/*****************************************************************************
 * Copyright (C) 2013-2020 MulticoreWare, Inc
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02111, USA.
 *****************************************************************************/

/*
 * cunn.h — Small MLP for CU split-recursion early-termination prediction.
 *
 * Architecture: 8 inputs → 16 hidden (ReLU) → 1 output (sigmoid)
 *
 * Feature vector:
 *   [0] cuVariance / 5000       (clamped 0-1, texture complexity from lowres)
 *   [1] depth / 3               (CU depth, 0 = 64x64 … 3 = 8x8)
 *   [2] (qp - 12) / 39          (quantisation parameter, normalised)
 *   [3] sliceType / 2           (0 = I, 0.5 = P, 1.0 = B)
 *   [4] lumaMean / maxPixel     (average luma value)
 *   [5] MAD / (lumaMean + 1)    (mean-absolute-deviation ratio — KEY feature)
 *   [6] minTempDepth / 3        (min depth from co-located reference CTU)
 *   [7] log(1 + sa8dCost) / 30  (merge/skip prediction-cost estimate)
 *
 * Output > 0.5  →  skip recursion (block is sufficiently uniform)
 * Output ≤ 0.5  →  recurse (block may benefit from finer partitioning)
 *
 * Default weights approximate complexityCheckCU(RDCOST_BASED_RSKIP):
 *   Neuron 0 implements ReLU(5 - 50*x[5]):
 *     x[5] = 0.00 → sigmoid(+9) ≈ 1.00  → skip   ✓
 *     x[5] = 0.09 → sigmoid( 0) = 0.50  → borderline (threshold ≈ 0.09)
 *     x[5] = 0.10 → sigmoid(-1) ≈ 0.27  → recurse ✓
 *   Neurons 1-15 carry small diverse seeds for gradient-based fine-tuning.
 *
 * Training workflow:
 *   1. Encode with --rskip 1 (RDCOST baseline).
 *   2. Record per-CU split ground-truth labels using --analysis-save.
 *   3. Extract the 8 features above for every evaluated CU.
 *   4. Train an 8→16→1 MLP (binary cross-entropy loss); export weights.
 *   5. Set X265_NN_WEIGHTS=/path/to/weights.bin and --rskip 3.
 *
 * Weight file format (raw float32 little-endian, 161 values = 644 bytes):
 *   W1[16][8]  b1[16]  W2[16]  b2
 */

#ifndef X265_CUNN_H
#define X265_CUNN_H

#include <math.h>
#include <stdio.h>

namespace X265_NS {

#define CUNN_INPUTS  8
#define CUNN_HIDDEN  16

/* ---------------------------------------------------------------------------
 * Weights — only this translation unit defines them (included once, from
 * analysis.cpp).  They are static so multiple TU inclusion is safe.
 * --------------------------------------------------------------------------- */

/* Layer 1 weights: W1[CUNN_HIDDEN][CUNN_INPUTS] */
static float g_cunnW1[CUNN_HIDDEN][CUNN_INPUTS] = {
    /* neuron 0: homo/mean threshold (approximates complexityCheckCU) */
    {  0.00f,  0.00f,  0.00f,  0.00f,  0.00f, -50.0f,  0.00f,  0.00f },
    /* neurons 1-15: small diverse seeds for gradient-based fine-tuning */
    {  0.01f,  0.01f,  0.01f,  0.01f,  0.01f,  0.01f,  0.01f,  0.01f },
    { -0.01f,  0.01f, -0.01f,  0.01f, -0.01f,  0.01f, -0.01f,  0.01f },
    {  0.02f, -0.01f,  0.01f, -0.02f,  0.01f, -0.01f,  0.02f, -0.01f },
    {  0.01f,  0.02f, -0.01f,  0.01f, -0.02f,  0.01f,  0.01f, -0.02f },
    { -0.02f,  0.01f,  0.01f, -0.01f,  0.02f, -0.01f, -0.01f,  0.02f },
    {  0.01f, -0.02f,  0.02f,  0.01f, -0.01f, -0.02f,  0.01f,  0.01f },
    {  0.02f,  0.01f, -0.02f, -0.01f,  0.01f,  0.02f, -0.01f, -0.02f },
    { -0.01f, -0.01f,  0.01f,  0.02f,  0.01f, -0.02f, -0.01f,  0.01f },
    {  0.01f,  0.02f, -0.01f, -0.02f, -0.01f,  0.01f,  0.02f, -0.01f },
    { -0.01f,  0.01f,  0.02f,  0.01f, -0.02f, -0.01f,  0.01f,  0.02f },
    {  0.02f, -0.02f,  0.01f, -0.01f,  0.01f,  0.02f, -0.02f,  0.01f },
    {  0.01f,  0.01f, -0.02f,  0.02f,  0.01f, -0.01f, -0.01f,  0.02f },
    { -0.02f,  0.01f,  0.01f, -0.01f, -0.02f,  0.01f,  0.02f, -0.01f },
    {  0.01f, -0.01f, -0.01f,  0.02f, -0.01f,  0.01f,  0.01f, -0.02f },
    {  0.02f,  0.01f, -0.01f, -0.01f,  0.02f, -0.01f, -0.02f,  0.01f },
};

/* Layer 1 biases: b1[CUNN_HIDDEN] */
static float g_cunnB1[CUNN_HIDDEN] = {
    5.0f,                                           /* neuron 0 threshold */
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
};

/* Layer 2 weights: W2[CUNN_HIDDEN] (output is scalar) */
static float g_cunnW2[CUNN_HIDDEN] = {
    2.0f,                                           /* only neuron 0 active */
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
};

/* Layer 2 bias (scalar) */
static float g_cunnB2 = -1.0f;

/* ---------------------------------------------------------------------------
 * cunnLoadWeights — load trained weights from a binary file.
 * File layout: W1[16][8] + b1[16] + W2[16] + b2  (161 float32 = 644 bytes).
 * Returns true on success; the default weights remain unchanged on failure.
 * --------------------------------------------------------------------------- */
static bool cunnLoadWeights(const char* path)
{
    FILE* f = fopen(path, "rb");
    if (!f)
        return false;

    bool ok = true;
    ok &= fread(g_cunnW1, sizeof(float), CUNN_HIDDEN * CUNN_INPUTS, f) == CUNN_HIDDEN * CUNN_INPUTS;
    ok &= fread(g_cunnB1, sizeof(float), CUNN_HIDDEN,               f) == (size_t)CUNN_HIDDEN;
    ok &= fread(g_cunnW2, sizeof(float), CUNN_HIDDEN,               f) == (size_t)CUNN_HIDDEN;
    ok &= fread(&g_cunnB2, sizeof(float), 1,                        f) == 1;
    fclose(f);
    return ok;
}

/* ---------------------------------------------------------------------------
 * cunnPredict — forward pass.
 * features[CUNN_INPUTS] → P(skip recursion) in (0, 1).
 * --------------------------------------------------------------------------- */
static float cunnPredict(const float* features)
{
    float hidden[CUNN_HIDDEN];

    /* Layer 1: linear + ReLU */
    for (int i = 0; i < CUNN_HIDDEN; i++)
    {
        float s = g_cunnB1[i];
        for (int j = 0; j < CUNN_INPUTS; j++)
            s += g_cunnW1[i][j] * features[j];
        hidden[i] = s > 0.f ? s : 0.f;
    }

    /* Layer 2: linear + sigmoid */
    float out = g_cunnB2;
    for (int i = 0; i < CUNN_HIDDEN; i++)
        out += g_cunnW2[i] * hidden[i];

    return 1.f / (1.f + expf(-out));
}

/* ---------------------------------------------------------------------------
 * Optional data-collection support (compile with -DX265_CUNN_COLLECT).
 *
 * Usage:
 *   cmake <src> -DCMAKE_CXX_FLAGS="-DX265_CUNN_COLLECT" ...
 *   X265_CUNN_LOG=cu_splits.csv  x265 input.yuv -o /dev/null \
 *       --preset medium --rskip 0
 *
 * IMPORTANT: use --rskip 0 so that both the split branch and the
 * no-split branch are always evaluated, giving unbiased ground-truth labels.
 *
 * CSV columns (raw, un-normalised values):
 *   cu_variance, depth, qp, slice_type, luma_mean, mad,
 *   min_temp_depth, sa8d_cost, split
 *
 * Then train:
 *   python3 tools/cunn_trainer.py train --data cu_splits.csv --out weights.bin
 * --------------------------------------------------------------------------- */
#ifdef X265_CUNN_COLLECT

static FILE* s_cunnLogFile = NULL;

/* Write one raw-feature row to the CSV log. Thread-safe on POSIX
 * (each fprintf to a shared stream is atomic per the C standard).    */
static void cunnLogCU(const uint64_t raw[8], int split_label)
{
    if (!s_cunnLogFile)
        return;
    fprintf(s_cunnLogFile,
            "%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%d\n",
            (unsigned long long)raw[0], (unsigned long long)raw[1],
            (unsigned long long)raw[2], (unsigned long long)raw[3],
            (unsigned long long)raw[4], (unsigned long long)raw[5],
            (unsigned long long)raw[6], (unsigned long long)raw[7],
            split_label);
}

#endif /* X265_CUNN_COLLECT */

} // namespace X265_NS

#endif // X265_CUNN_H
