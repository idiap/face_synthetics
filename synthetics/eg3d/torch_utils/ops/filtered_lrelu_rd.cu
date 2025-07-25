/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaSourceCode1WayCommercial
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "filtered_lrelu.cu"

// Template/kernel specializations for sign read mode.

// Full op, 32-bit indexing.
template filtered_lrelu_kernel_spec choose_filtered_lrelu_kernel<c10::Half, int32_t, false, true>(const filtered_lrelu_kernel_params& p, int sharedKB);
template filtered_lrelu_kernel_spec choose_filtered_lrelu_kernel<float,     int32_t, false, true>(const filtered_lrelu_kernel_params& p, int sharedKB);

// Full op, 64-bit indexing.
template filtered_lrelu_kernel_spec choose_filtered_lrelu_kernel<c10::Half, int64_t, false, true>(const filtered_lrelu_kernel_params& p, int sharedKB);
template filtered_lrelu_kernel_spec choose_filtered_lrelu_kernel<float,     int64_t, false, true>(const filtered_lrelu_kernel_params& p, int sharedKB);

// Activation/signs only for generic variant. 64-bit indexing.
template void* choose_filtered_lrelu_act_kernel<c10::Half, false, true>(void);
template void* choose_filtered_lrelu_act_kernel<float,     false, true>(void);
template void* choose_filtered_lrelu_act_kernel<double,    false, true>(void);

// Copy filters to constant memory.
template cudaError_t copy_filters<false, true>(cudaStream_t stream);
