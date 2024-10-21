//===----------------------------------------------------------------------===//
//
// Copyright (C) 2023 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <cstdlib>
#include <vector>
#include <assert.h>
#include <chrono>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "memory.h"
#include "bmruntime_interface.h"
#include <getopt.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <random>
#include <numeric>

namespace py = pybind11;

static const uint16_t ATTENTION_MASK = 0xF0E2;

class TTSLlama
{
public:
  void init(const std::vector<int> &devid, std::string model_path);
  void deinit();

  int forward_first_text(std::vector<int> &tokens);
  int forward_next_text();
  // std::vector<float> forward_first_text_core(std::vector<int> &tokens);
  // std::vector<float> forward_next_text_core(int newtoken);
  std::vector<int> generate_text(std::vector<int> &history_tokens, int EOS, float tempreature);

  std::pair<std::vector<int>, std::vector<uint16_t>> forward_first_code(std::vector<int> &tokens, int idx_spk, std::vector<float> spk_emb);
  std::pair<std::vector<int>, std::vector<uint16_t>> forward_next_code();
  py::dict generate_code(std::vector<int> &history_tokens, int spk_idx, std::vector<float> &spk_emb, int EOS, float tempreture);

  std::pair<std::vector<float>, std::vector<float>> forward_first_code_core(std::vector<int> &tokens, int idx_spk, std::vector<float> spk_emb);
  std::pair<std::vector<float>, std::vector<float>> forward_next_code_core(std::vector<int> &newtoken);
  union Fp32
  {
    uint32_t u;
    float f;
  };

  std::mt19937 sgen;
  TTSLlama() : sgen(std::random_device()()){};

private:
  void net_launch(const bm_net_info_t *net, int stage_idx = 0);
  inline void d2d(bm_device_mem_t &dst, bm_device_mem_t &src);

  void head_launch(const bm_net_info_t *net, bm_device_mem_t &logits_mem);

  int greedy_search_text(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  std::vector<int> greedy_search_code(const bm_net_info_t *net, bm_device_mem_t &logits_mem);

  int penalty_sample_text(const bm_net_info_t *net, bm_device_mem_t &logits_mem);
  std::vector<int> penalty_sample_code(const bm_net_info_t *net, bm_device_mem_t &logits_mem);

  // data type conversion
  inline float half2float(uint16_t value);
  inline uint16_t float2uint16(float value);

public:
  int text_token_length;
  int code_token_length;
  int new_code_token_length;
  int new_text_token_length;
  int hidden_size; // read from bmodel
  int SEQLEN;      // read from bmodel
  int NUM_LAYERS;  // read from bmodel
  int NUM_VQ;     // read from bmodel
  int AUDIO_CODE_SIZE; // read from bmodel
  bool io_alone;
  std::vector<int> visited_text_tokens;
  std::vector<std::vector<int>> visited_code_tokens;
  bool DEBUGGING = true;

  // generation
  int repeat_last_n;
  float temperature;
  float top_p;
  float repeat_penalty;
  int max_new_tokens;
  std::string generation_mode_text = "penalty_sample";
  std::string generation_mode_code = "penalty_sample";

private:
  std::vector<bm_handle_t> handles;
  bm_handle_t bm_handle;
  void *p_bmrt;
  std::vector<const bm_net_info_t *> net_blocks;
  std::vector<const bm_net_info_t *> net_blocks_cache;
  const bm_net_info_t *net_embed_text, *net_embed_code;
  const bm_net_info_t *net_embed_text_cache, *net_embed_code_cache;
  const bm_net_info_t *net_lm_text, *net_lm_code;
  const bm_net_info_t *net_penalty_sample_head_text, *net_penalty_sample_head_code;
  const bm_net_info_t *net_greedy_head_text, *net_greedy_head_code;
  std::vector<bm_device_mem_t> past_key;
  std::vector<bm_device_mem_t> past_value;
};

// ref: https://blog.csdn.net/qq_39016531/article/details/107411030
uint16_t TTSLlama::float2uint16(float value)
{

    const Fp32 f32infty = { 255U << 23 };
    const Fp32 f16infty = { 31U << 23 };
    const Fp32 magic = { 15U << 23 };
    const uint32_t sign_mask = 0x80000000U;
    const uint32_t round_mask = ~0xFFFU;

    Fp32 in;
    uint16_t out;

    in.f = value;

    uint32_t sign = in.u & sign_mask;
    in.u ^= sign;

    if (in.u >= f32infty.u) /* Inf or NaN (all exponent bits set) */
    {
        /* NaN->sNaN and Inf->Inf */
        out = (in.u > f32infty.u) ? 0x7FFFU : 0x7C00U;
    }
    else /* (De)normalized number or zero */
    {
        in.u &= round_mask;
        in.f *= magic.f;
        in.u -= round_mask;
        if (in.u > f16infty.u)
        {
            in.u = f16infty.u; /* Clamp to signed infinity if overflowed */
        }

        out = uint16_t(in.u >> 13); /* Take the bits! */
    }

    out = uint16_t(out | (sign >> 16));

    return out;
}

float TTSLlama::half2float(uint16_t value)
{
    const Fp32 magic = { (254U - 15U) << 23 };
    const Fp32 was_infnan = { (127U + 16U) << 23 };
    Fp32 out;

    out.u = (value & 0x7FFFU) << 13;   /* exponent/mantissa bits */
    out.f *= magic.f;                  /* exponent adjust */
    if (out.f >= was_infnan.f)         /* make sure Inf/NaN survive */
    {
        out.u |= 255U << 23;
    }
    out.u |= (value & 0x8000U) << 16;  /* sign bit */

    return out.f;
}

// float TTSLlama::half2float(uint16_t x) // fp16 -> fp32
// {
//     unsigned sign = ((x >> 15) & 1);
//     unsigned exponent = ((x >> 10) & 0x1f);
//     unsigned mantissa = ((x & 0x3ff) << 13);
//     if (exponent == 0x1f) {  /* NaN or Inf */
//         mantissa = (mantissa ? (sign = 0, 0x7fffff) : 0);
//         exponent = 0xff;
//     } else if (!exponent) {  /* Denorm or Zero */
//         if (mantissa) {
//             unsigned int msb;
//             exponent = 0x71;
//             do {
//                 msb = (mantissa & 0x400000);
//                 mantissa <<= 1;  /* normalize */
//                 --exponent;
//             } while (!msb);
//             mantissa &= 0x7fffff;  /* 1.mantissa is implicit */
//         }
//     } else {
//         exponent += 0x70;
//     }
//     int temp = ((sign << 31) | (exponent << 23) | mantissa);
 
//     return *((float*)((void*)&temp));
// }

void TTSLlama::net_launch(const bm_net_info_t *net, int stage_idx)
{
  std::vector<bm_tensor_t> in_tensors(net->input_num);
  std::vector<bm_tensor_t> out_tensors(net->output_num);

  for (int i = 0; i < net->input_num; i++)
  {
    bmrt_tensor_with_device(
        &in_tensors[i], net->stages[stage_idx].input_mems[i],
        net->input_dtypes[i], net->stages[stage_idx].input_shapes[i]);
  }
  for (int i = 0; i < net->output_num; i++)
  {
    bmrt_tensor_with_device(
        &out_tensors[i], net->stages[stage_idx].output_mems[i],
        net->output_dtypes[i], net->stages[stage_idx].output_shapes[i]);
  }
  auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);
  assert(ret);
  bm_thread_sync(bm_handle);
}

void TTSLlama::d2d(bm_device_mem_t &dst, bm_device_mem_t &src)
{
  bm_memcpy_d2d_byte(bm_handle, dst, 0, src, 0, bm_mem_get_device_size(src));
}

void TTSLlama::init(const std::vector<int> &devices, std::string model_path)
{

  // request bm_handle
  std::cout << "Device [ ";
  for (auto d : devices)
  {
    std::cout << d << " ";
  }
  std::cout << "] loading ....\n";
  for (auto d : devices)
  {
    bm_handle_t h;
    bm_status_t status = bm_dev_request(&h, d);
    assert(BM_SUCCESS == status);
    handles.push_back(h);
  }
  bm_handle = handles[0];

  // create bmruntime
#ifdef SOC_TARGET
  p_bmrt = bmrt_create(handles[0]);
#else
  p_bmrt = bmrt_create_ex(handles.data(), handles.size());
#endif
  assert(NULL != p_bmrt);

  // load bmodel by file
  printf("Model[%s] loading ....\n", model_path.c_str());
  bool ret = bmrt_load_bmodel(p_bmrt, model_path.c_str());
  assert(true == ret);
  printf("Done!\n");

  // net embed and lm_head
  net_embed_text = bmrt_get_network_info(p_bmrt, "embedding_text");
  net_embed_text_cache = bmrt_get_network_info(p_bmrt, "embedding_text_cache");
  net_embed_code = bmrt_get_network_info(p_bmrt, "embedding_code");
  net_embed_code_cache = bmrt_get_network_info(p_bmrt, "embedding_code_cache");

  net_lm_text = bmrt_get_network_info(p_bmrt, "lm_head_text");
  net_lm_code = bmrt_get_network_info(p_bmrt, "lm_head_code");

  net_penalty_sample_head_text = bmrt_get_network_info(p_bmrt, "penalty_sample_head_text");
  net_penalty_sample_head_code = bmrt_get_network_info(p_bmrt, "penalty_sample_head_code");

  net_greedy_head_text = bmrt_get_network_info(p_bmrt, "greedy_head_text");
  net_greedy_head_code = bmrt_get_network_info(p_bmrt, "greedy_head_code");

  SEQLEN = net_embed_text->stages[0].input_shapes[0].dims[1]; // real seqlen
  printf("SEQLEN: %d\n", SEQLEN);
  auto num_nets = bmrt_get_network_number(p_bmrt);
  printf("num_nets: %d\n", num_nets);
  NUM_LAYERS = 20; // (num_nets - 10) / 2;
  hidden_size = net_embed_text->stages[0].output_shapes[0].dims[2];
  NUM_VQ = net_lm_code->stages[0].output_shapes[0].dims[2];
  AUDIO_CODE_SIZE = net_lm_code->stages[0].output_shapes[0].dims[1];
  printf("NUM_LAYERS: %d, hidden_size: %d, NUM_VQ: %d\n", NUM_LAYERS, hidden_size, NUM_VQ);

  // resize
  visited_text_tokens.resize(SEQLEN);
  visited_code_tokens.resize(SEQLEN);
  for (auto &inner_vec : visited_code_tokens)
  {
    inner_vec.resize(NUM_VQ);
  }

  // net blocks
  for (int i = 0; i < NUM_LAYERS; i++)
  {
    auto block_name = "block_" + std::to_string(i);
    auto cache_name = "block_cache_" + std::to_string(i);
    net_blocks.emplace_back(bmrt_get_network_info(p_bmrt, block_name.c_str()));
    net_blocks_cache.emplace_back(
        bmrt_get_network_info(p_bmrt, cache_name.c_str()));
  }

  // kv cache
  past_key.resize(NUM_LAYERS);
  past_value.resize(NUM_LAYERS);
  auto addr_mode = net_blocks_cache[0]->addr_mode;
  io_alone = addr_mode == 1;
  for (int i = 0; i < NUM_LAYERS; i++)
  {
    assert(addr_mode == net_blocks_cache[i]->addr_mode);
    if (io_alone)
    {
      past_key[i] = net_blocks_cache[i]->stages[0].input_mems[3];
      past_value[i] = net_blocks_cache[i]->stages[0].input_mems[4];
    }
    else
    {
      auto ret = bm_malloc_device_byte(bm_handle, &past_key[i],
                                       net_blocks_cache[i]->max_input_bytes[3]);
      assert(BM_SUCCESS == ret);
      ret = bm_malloc_device_byte(bm_handle, &past_value[i],
                                  net_blocks_cache[i]->max_input_bytes[4]);
      assert(BM_SUCCESS == ret);
    }
  }
}

void TTSLlama::deinit()
{
  if (false == io_alone)
  {
    for (int i = 0; i < NUM_LAYERS; i++)
    {
      bm_free_device(bm_handle, past_key[i]);
      bm_free_device(bm_handle, past_value[i]);
    }
  }
  bmrt_destroy(p_bmrt);
  for (auto h : handles)
  {
    bm_dev_free(h);
  }
}

void TTSLlama::head_launch(const bm_net_info_t *net, bm_device_mem_t &logits_mem)
{
  std::vector<bm_tensor_t> in_tensors(net->input_num);
  std::vector<bm_tensor_t> out_tensors(net->output_num);

  bmrt_tensor_with_device(                                   // #############
      &in_tensors[0], logits_mem,                            // size = 84712 21178*4 for text generation;
      net->input_dtypes[0], net->stages[0].input_shapes[0]); //

  for (int i = 1; i < net->input_num; i++)
  {
    bmrt_tensor_with_device(
        &in_tensors[i], net->stages[0].input_mems[i],
        net->input_dtypes[i], net->stages[0].input_shapes[i]);
  }
  for (int i = 0; i < net->output_num; i++)
  {
    bmrt_tensor_with_device(
        &out_tensors[i], net->stages[0].output_mems[i],
        net->output_dtypes[i], net->stages[0].output_shapes[i]);
  }
  auto ret = bmrt_launch_tensor_ex(p_bmrt, net->name, in_tensors.data(),
                                   net->input_num, out_tensors.data(),
                                   net->output_num, true, false);
  assert(ret);
  bm_thread_sync(bm_handle);
}

int TTSLlama::greedy_search_text(const bm_net_info_t *net, bm_device_mem_t &logits_mem) {
  auto &out_mem = net->stages[0].output_mems[0];
  head_launch(net, logits_mem);
  int token = 0;
  bm_memcpy_d2s(bm_handle, (void *)&token, out_mem);
  return token;
}

int TTSLlama::penalty_sample_text(const bm_net_info_t *net, bm_device_mem_t &logits_mem)
{
  auto &in1_mem = net->stages[0].input_mems[1];
  auto &in2_mem = net->stages[0].input_mems[2];
  auto &in3_mem = net->stages[0].input_mems[3];
  auto &in4_mem = net->stages[0].input_mems[4];
  auto &out0_mem = net->stages[0].output_mems[0];
  auto &out1_mem = net->stages[0].output_mems[1];

  // repeat_penalty + top_p + top_k + temperature
  std::vector<int> generated_tokens(SEQLEN, visited_text_tokens[text_token_length - 1]); // #####
  repeat_last_n = std::min(repeat_last_n, text_token_length); //################
  std::copy(visited_text_tokens.begin() + text_token_length - repeat_last_n,
            visited_text_tokens.begin() + text_token_length,
            generated_tokens.begin());
  bm_memcpy_s2d(bm_handle, in1_mem, (void *)generated_tokens.data());
  bm_memcpy_s2d(bm_handle, in2_mem, (void *)&top_p);
  bm_memcpy_s2d(bm_handle, in3_mem, (void *)&temperature);
  bm_memcpy_s2d(bm_handle, in4_mem, (void *)&repeat_penalty);

  // inference
  head_launch(net, logits_mem);

  // get logit & token
  int candidate_num = net->stages[0].output_shapes[0].dims[1];
  std::vector<float> probs(candidate_num);
  bm_memcpy_d2s(bm_handle, probs.data(), out0_mem);
  std::vector<int> tokens(candidate_num);
  bm_memcpy_d2s(bm_handle, tokens.data(), out1_mem);

  // penalty_sample
  std::discrete_distribution<> dist(probs.begin(), probs.end());
  return tokens[dist(sgen)];
}

int TTSLlama::forward_first_text(std::vector<int> &tokens)
{
  std::vector<int> position_id(SEQLEN, 0);
  std::vector<uint16_t> attention_mask(SEQLEN * SEQLEN, ATTENTION_MASK);
  std::copy(tokens.begin(), tokens.end(), visited_text_tokens.data());

  text_token_length = tokens.size();

  for (int i = 0; i < text_token_length; i++)
  {
    position_id[i] = i;
  }
  for (int i = 0; i < text_token_length; i++)
  {
    for (int j = 0; j < SEQLEN; j++)
    {
      if (j <= i)
      {
        attention_mask[i * SEQLEN + j] = 0;
      }
    }
  }

  // forward embeding
  auto &in_mem = net_embed_text->stages[0].input_mems[0];
  auto &out_mem = net_embed_text->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)visited_text_tokens.data());
  net_launch(net_embed_text); // prefil embedding

  // forward blocks
  for (int idx = 0; idx < NUM_LAYERS; idx++)
  {
    auto &in0_mem = net_blocks[idx]->stages[0].input_mems[0];
    auto &in1_mem = net_blocks[idx]->stages[0].input_mems[1];
    auto &in2_mem = net_blocks[idx]->stages[0].input_mems[2];
    d2d(in0_mem, out_mem);
    if (idx == 0)
    {
      // only first time need copy
      bm_memcpy_s2d(bm_handle, in1_mem, (void *)position_id.data());
      bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
    }
    net_launch(net_blocks[idx]);
    out_mem = net_blocks[idx]->stages[0].output_mems[0];
    d2d(past_key[idx], net_blocks[idx]->stages[0].output_mems[1]);
    d2d(past_value[idx], net_blocks[idx]->stages[0].output_mems[2]);
    
    // debug
    if(DEBUGGING) {
      std::vector<uint16_t> temp(SEQLEN * hidden_size, 0);
      bm_memcpy_d2s(bm_handle, temp.data(), out_mem);
      printf("forward first text layer %d\n", idx);
      for (int i = 0; i < hidden_size; i++)
      {
        if(i%5 == 0) printf("\n");
        printf("%.04f ", half2float(temp[(text_token_length-1)*hidden_size + i]));
      }
    }
  }

  // forward lmhead
  int bytes = out_mem.size / SEQLEN;
  auto &lm_in_mem = net_lm_text->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm_text->stages[0].output_mems[0];
  bm_memcpy_d2d_byte(bm_handle, lm_in_mem, 0, out_mem,
                     (text_token_length - 1) * bytes, bytes);
  net_launch(net_lm_text);

  int token = 0;
  if (generation_mode_text == "greedy") {
    token = greedy_search_text(net_greedy_head_text, lm_out_mem);
  }
  else if (generation_mode_text == "penalty_sample") {
    token = penalty_sample_text(net_penalty_sample_head_text, lm_out_mem); // size = 84712
  }
  visited_text_tokens[text_token_length] = token;
  text_token_length += 1;
  return token;
}

int TTSLlama::forward_next_text()
{
  int cur_token = visited_text_tokens[text_token_length - 1];

  std::vector<uint16_t> attention_mask(SEQLEN + 1, 0);
  for (int i = text_token_length - 1; i < SEQLEN; i++)
  {
    attention_mask[i] = ATTENTION_MASK;
  }
  int32_t position_id = text_token_length - 1;

  // embedding
  auto &in_mem = net_embed_text_cache->stages[0].input_mems[0];
  auto &out_mem = net_embed_text_cache->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)&cur_token);
  net_launch(net_embed_text_cache);

  // blocks
  int bytes =
      bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[1]);
  int token_offset = (text_token_length - 1) * bytes;
  for (int idx = 0; idx < NUM_LAYERS; idx++)
  {
    auto &in0_mem = net_blocks_cache[idx]->stages[0].input_mems[0];
    auto &in1_mem = net_blocks_cache[idx]->stages[0].input_mems[1];
    auto &in2_mem = net_blocks_cache[idx]->stages[0].input_mems[2];
    auto &in3_mem = net_blocks_cache[idx]->stages[0].input_mems[3];
    auto &in4_mem = net_blocks_cache[idx]->stages[0].input_mems[4];
    auto &out0_mem = net_blocks_cache[idx]->stages[0].output_mems[0];
    auto &out1_mem = net_blocks_cache[idx]->stages[0].output_mems[1];
    auto &out2_mem = net_blocks_cache[idx]->stages[0].output_mems[2];
    d2d(in0_mem, out_mem);
    if (io_alone)
    {
      if (idx == 0)
      {
        bm_memcpy_s2d(bm_handle, in1_mem, (void *)&position_id);
        bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
      }
      else
      {
        d2d(in1_mem, net_blocks_cache[0]->stages[0].input_mems[1]);
        d2d(in2_mem, net_blocks_cache[0]->stages[0].input_mems[2]);
      }
    }
    else
    {
      if (idx == 0)
      {
        bm_memcpy_s2d(bm_handle, in1_mem, (void *)&position_id);
        bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
      }
      d2d(in3_mem, past_key[idx]);
      d2d(in4_mem, past_value[idx]);
    }
    net_launch(net_blocks_cache[idx]);
    out_mem = out0_mem;
    bm_memcpy_d2d_byte(bm_handle, past_key[idx], token_offset, out1_mem, 0,
                       bytes);
    bm_memcpy_d2d_byte(bm_handle, past_value[idx], token_offset, out2_mem, 0,
                       bytes);
    
    // debug
    if(DEBUGGING) {
      std::vector<uint16_t> temp(hidden_size, 0);
      bm_memcpy_d2s(bm_handle, temp.data(), out0_mem);
      printf("\nforward next text layer %d\n", idx);
      for (int i = 0; i < hidden_size; i++)
      {
        if(i%5 == 0) printf("\n");
        printf("%.04f ", half2float(temp[i]));
      }
    }
  }

  // forward lmhead
  auto &lm_in_mem = net_lm_text->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm_text->stages[0].output_mems[0];
  d2d(lm_in_mem, out_mem);
  net_launch(net_lm_text);

  int token = 0;
  if (generation_mode_text == "greedy") {
    token = greedy_search_text(net_greedy_head_text, lm_out_mem);
  }
  else if (generation_mode_text == "penalty_sample") {
    token = penalty_sample_text(net_penalty_sample_head_text, lm_out_mem); // size = 84712
  }

  visited_text_tokens[text_token_length] = token;
  text_token_length += 1;
  return token;
}

std::vector<int> TTSLlama::generate_text(std::vector<int> &history_tokens, int EOS, float temp_tempreature)
{
  printf("generate text\n");
  if (history_tokens.empty())
  {
    printf("Sorry: your text is empty!!\n");
    history_tokens.clear();
    return {};
  }

  // make sure token not too large
  if ((int)history_tokens.size() > SEQLEN - 10)
  {
    history_tokens.clear();
    printf("Error: your original text is too large!\n");
    return {};
  }
  std::fill(visited_text_tokens.begin(), visited_text_tokens.end(), 0);
  temperature = temp_tempreature;

  new_text_token_length = 0;
  std::vector<int> result_tokens;
  int token = forward_first_text(history_tokens);
  while (token != EOS && text_token_length < SEQLEN)
  {
    result_tokens.emplace_back(token);
    new_text_token_length += 1;
    token = forward_next_text();
  }
  return result_tokens;
}

std::vector<int> TTSLlama::greedy_search_code(const bm_net_info_t *net, bm_device_mem_t &logits_mem) {
  auto &out_mem = net->stages[0].output_mems[0];
  head_launch(net, logits_mem);
  std::vector<int> token(NUM_VQ, 0);
  bm_memcpy_d2s(bm_handle, (void *)&token, out_mem);
  return token;
}

std::vector<int> TTSLlama::penalty_sample_code(const bm_net_info_t *net, bm_device_mem_t &logits_mem)
{
  auto &in1_mem = net->stages[0].input_mems[1];
  auto &in2_mem = net->stages[0].input_mems[2];
  auto &in3_mem = net->stages[0].input_mems[3];
  auto &in4_mem = net->stages[0].input_mems[4];
  auto &out0_mem = net->stages[0].output_mems[0];
  auto &out1_mem = net->stages[0].output_mems[1];

  // repeat_penalty + top_p + top_k + temperature
  std::vector<int> generated_tokens(SEQLEN * NUM_VQ, 625); // ##### visited_tokens[token_length - 1]

  // for(int i = 0; i < SEQLEN; i++)
  // {
  //   for(int j = 0; j < NUM_VQ; j++)
  //   {
  //     generated_tokens[i * NUM_VQ + j] = visited_code_tokens[code_token_length - 1][j];
  //   }
  // }
  
  // if(DEBUGGING) 
  // {
  //   printf("generated_tokens = \n");
  //   for (int i = 0; i < int(generated_tokens.size()); ++i) 
  //   {
  //     if(i%5 == 0) printf("\n");
  //     printf("%d ", generated_tokens[i]);
  //   }
  // }

  repeat_last_n = std::min(repeat_last_n, code_token_length);

  for (int i = 0; i < repeat_last_n; ++i) 
  {
      std::copy(visited_code_tokens[code_token_length - repeat_last_n + i].begin(), visited_code_tokens[code_token_length - repeat_last_n + i].end(), generated_tokens.begin() + i * NUM_VQ);
  }
  if(DEBUGGING) 
  {
    printf("repeat_last_n = %d\n", repeat_last_n);
    printf("generated_tokens = \n");
    for (int i = 0; i < int(generated_tokens.size()); ++i) 
    {
      if(i%5 == 0) printf("\n");
      printf("%d ", generated_tokens[i]);
    }
  }
  bm_memcpy_s2d(bm_handle, in1_mem, (void *)generated_tokens.data());
  bm_memcpy_s2d(bm_handle, in2_mem, (void *)&top_p);
  bm_memcpy_s2d(bm_handle, in3_mem, (void *)&temperature);
  bm_memcpy_s2d(bm_handle, in4_mem, (void *)&repeat_penalty);

  // inference
  head_launch(net, logits_mem);

  // get logit & token
  int candidate_num = net->stages[0].output_shapes[0].dims[2];
  std::vector<float> probs_flat(NUM_VQ * candidate_num);
  std::vector<float> tokens_flat(NUM_VQ * candidate_num); // caution: the output tokens of penalty_sample_head_code is [float] not [int]

  bm_memcpy_d2s(bm_handle, probs_flat.data(), out0_mem);
  bm_memcpy_d2s(bm_handle, tokens_flat.data(), out1_mem);

  std::vector<int> curr_token(NUM_VQ, 0);
  // penalty_sample
  for(int i = 0; i < NUM_VQ; i++)
  {
    std::discrete_distribution<> dist(probs_flat.begin()+i*candidate_num, probs_flat.begin()+(i+1)*candidate_num);
    curr_token[i] = int(tokens_flat[i*candidate_num + dist(sgen)]);
  }
  return curr_token;
}

std::pair<std::vector<int>, std::vector<uint16_t>> TTSLlama::forward_first_code(std::vector<int> &tokens, int idx_spk, std::vector<float> spk_emb)
{
  printf("forward_first_code\n");
  std::vector<int> position_id(SEQLEN, 0);
  std::vector<uint16_t> attention_mask(SEQLEN * SEQLEN, ATTENTION_MASK);
  
  for(int i = 0; i < int(tokens.size()); i++) 
  {
    std::fill(visited_code_tokens[i].begin(), visited_code_tokens[i].end(), tokens[i]);
  }

  code_token_length = tokens.size();

  for (int i = 0; i < code_token_length; i++)
  {
    position_id[i] = i;
  }
  for (int i = 0; i < code_token_length; i++)
  {
    for (int j = 0; j < SEQLEN; j++)
    {
      if (j <= i)
      {
        attention_mask[i * SEQLEN + j] = 0;
      }
    }
  }

  // forward embeding
  // i == 0, use text_emb
  tokens.resize(SEQLEN, 0);
  auto &in_mem = net_embed_text->stages[0].input_mems[0]; 
  auto &out_mem = net_embed_text->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)tokens.data()); // i == 0##########
  printf("token_length: %d\n", code_token_length);
  net_launch(net_embed_text); // prefil embedding

  if(DEBUGGING) {
    std::vector<uint16_t> tmp_emb(SEQLEN*hidden_size, 0); // debug
    bm_memcpy_d2s(bm_handle, tmp_emb.data(), out_mem); // debug
    // debug
    for(int t = 0; t < code_token_length; t++) {
      for (int i = 0; i < hidden_size; i++)
      {
        if(i%5 == 0) printf("\n");
        printf("%.04f ", half2float(tmp_emb[t*hidden_size + i]));
      }
      printf("================== \n");
    }
  }
  
  // the embedding of empty_spk --> spk_emb
  if (idx_spk != -1)
  {
    std::vector<uint16_t> tmp(hidden_size, 0);
    for(int i = 0; i < hidden_size; i++)
    {
      tmp[i] = float2uint16(spk_emb[i]);
    }
    int byte_size = out_mem.size / hidden_size;
    bm_memcpy_s2d_partial_offset(bm_handle, out_mem, (void *)tmp.data(), hidden_size * byte_size, idx_spk * hidden_size * byte_size); // bytesize, offset
  }

  // forward blocks
  for (int idx = 0; idx < NUM_LAYERS; idx++)
  {
    auto &in0_mem = net_blocks[idx]->stages[0].input_mems[0];
    auto &in1_mem = net_blocks[idx]->stages[0].input_mems[1];
    auto &in2_mem = net_blocks[idx]->stages[0].input_mems[2];
    d2d(in0_mem, out_mem);
    if (idx == 0)
    {
      // only first time need copy
      bm_memcpy_s2d(bm_handle, in1_mem, (void *)position_id.data());
      bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
    }
    net_launch(net_blocks[idx]);
    out_mem = net_blocks[idx]->stages[0].output_mems[0]; // hiddens
    
    d2d(past_key[idx], net_blocks[idx]->stages[0].output_mems[1]);
    d2d(past_value[idx], net_blocks[idx]->stages[0].output_mems[2]);

    // debug
    if(DEBUGGING) {
      std::vector<uint16_t> temp(hidden_size*SEQLEN, 0);
      bm_memcpy_d2s(bm_handle, temp.data(), out_mem);
      printf("\nforward first code layer %d\n", idx);
      for (int i = 0; i < hidden_size; i++)
      {
        if(i%5 == 0) printf("\n");
        printf("%.04f ", half2float(temp[(code_token_length-1)*hidden_size + i]));
      }
    }
  }
  // (bm_handle, past_value[idx], token_offset, out2_mem, 0, bytes);
  // forward lmhead
  int bytes = out_mem.size / SEQLEN;
  auto &lm_in_mem = net_lm_code->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm_code->stages[0].output_mems[0];
  bm_memcpy_d2d_byte(bm_handle, lm_in_mem, 0, out_mem,
                     (code_token_length - 1) * bytes, bytes);

  if(DEBUGGING) { 
    printf("\nBefore lmhead\n"); // bug here
    std::vector<uint16_t> temp0(hidden_size, 0);
    bm_memcpy_d2s(bm_handle, temp0.data(), lm_in_mem);
    for (int i = 0; i < int(temp0.size()); i++)
    {
      if(i%5 == 0) printf("\n");
      printf("%.04f ", half2float(temp0[i]));
    }
  }

  net_launch(net_lm_code); // B

  // debug
  if(DEBUGGING) { 
    std::vector<float> temp(626*4, 0);
    bm_memcpy_d2s(bm_handle, temp.data(), lm_out_mem);
    printf("\nAfter lmhead\n");
    for (int i = 0; i < int(temp.size()); i++)
    {
      if (i % 626 == 0) printf(" ====================== ");
      if(i%5 == 0) printf("\n");
      printf("%.04f ", temp[i]);
    }
  }

  std::vector<uint16_t> last_hidden(hidden_size, 0);
  bm_memcpy_d2s_partial_offset(bm_handle, (void *)last_hidden.data(), out_mem, bytes, (code_token_length - 1) * bytes); // ##### 1*512*12*64*[2B]
  if(DEBUGGING) {
    printf("\nlast hidden\n"); 
    for (int i = 0; i < int(last_hidden.size()); i++) 
    { printf("%.04f ", half2float(last_hidden[i])); }
    printf("\n");
  }  

  std::vector<int> token;
  code_token_length = 0;
  if (generation_mode_code == "greedy") {
    token = greedy_search_code(net_greedy_head_code, lm_out_mem);
  } else if (generation_mode_code == "penalty_sample") {
    token = penalty_sample_code(net_penalty_sample_head_code, lm_out_mem);
  } else {
    printf("Error: unknown generation mode: %s\n", generation_mode_code.c_str());
  }
  visited_code_tokens[0] = token;
  code_token_length = 1;
  std::pair<std::vector<int>, std::vector<uint16_t>> result;
  result.first = token;
  result.second = last_hidden;
  return result;
}

std::pair<std::vector<int>, std::vector<uint16_t>> TTSLlama::forward_next_code()
{
  std::vector<int> cur_token = visited_code_tokens[code_token_length - 1];

  std::vector<uint16_t> attention_mask(SEQLEN + 1, 0);
  for (int i = code_token_length - 1; i < SEQLEN; i++)
  {
    attention_mask[i] = ATTENTION_MASK;
  }
  int32_t position_id = code_token_length - 1;

  // embedding
  auto &in_mem = net_embed_code_cache->stages[0].input_mems[0];
  auto &out_mem = net_embed_code_cache->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)cur_token.data());
  net_launch(net_embed_code_cache);

  if (DEBUGGING) {
    std::vector<int> tmp(4);
    bm_memcpy_d2s(bm_handle, tmp.data(), in_mem);
    printf("\nnet_embed_code_cache in\n");
    for (int i = 0; i < 4; i++)
    {
      printf("%d  ", tmp[i]);
    }
    std::vector<uint16_t> emb_out(hidden_size);
    bm_memcpy_d2s(bm_handle, emb_out.data(), out_mem);
    printf("\nnet_embed_code_cache out\n");
    for (int i = 0; i < hidden_size; i++)
    {
      if(i%5 == 0) printf("\n");
      printf("%.04f ", half2float(emb_out[i]));
    }
  }
  // blocks
  int bytes =
      bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[1]);
  int token_offset = (code_token_length - 1) * bytes;
  for (int idx = 0; idx < NUM_LAYERS; idx++)
  {
    auto &in0_mem = net_blocks_cache[idx]->stages[0].input_mems[0];
    auto &in1_mem = net_blocks_cache[idx]->stages[0].input_mems[1];
    auto &in2_mem = net_blocks_cache[idx]->stages[0].input_mems[2];
    auto &in3_mem = net_blocks_cache[idx]->stages[0].input_mems[3];
    auto &in4_mem = net_blocks_cache[idx]->stages[0].input_mems[4];
    auto &out0_mem = net_blocks_cache[idx]->stages[0].output_mems[0];
    auto &out1_mem = net_blocks_cache[idx]->stages[0].output_mems[1];
    auto &out2_mem = net_blocks_cache[idx]->stages[0].output_mems[2];
    d2d(in0_mem, out_mem);
    if (io_alone)
    {
      if (idx == 0)
      {
        bm_memcpy_s2d(bm_handle, in1_mem, (void *)&position_id);
        bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
      }
      else
      {
        d2d(in1_mem, net_blocks_cache[0]->stages[0].input_mems[1]);
        d2d(in2_mem, net_blocks_cache[0]->stages[0].input_mems[2]);
      }
    }
    else
    {
      if (idx == 0)
      {
        bm_memcpy_s2d(bm_handle, in1_mem, (void *)&position_id);
        bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
      }
      d2d(in3_mem, past_key[idx]);
      d2d(in4_mem, past_value[idx]);
    }
    net_launch(net_blocks_cache[idx]);
    out_mem = out0_mem;
    bm_memcpy_d2d_byte(bm_handle, past_key[idx], token_offset, out1_mem, 0,
                       bytes);
    bm_memcpy_d2d_byte(bm_handle, past_value[idx], token_offset, out2_mem, 0,
                       bytes);
  }

  std::vector<uint16_t> last_hidden(hidden_size, 0);  // debug
  bm_memcpy_d2s(bm_handle, (void *)last_hidden.data(), out_mem);

  // forward lmhead
  auto &lm_in_mem = net_lm_code->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm_code->stages[0].output_mems[0];
  d2d(lm_in_mem, out_mem);
  net_launch(net_lm_code);

  std::vector<int> token(NUM_VQ, 0);

  if (generation_mode_code == "greedy") {
    token = greedy_search_code(net_greedy_head_code, lm_out_mem);
  } else if (generation_mode_code == "penalty_sample") {
    token = penalty_sample_code(net_penalty_sample_head_code, lm_out_mem);
  }

  visited_code_tokens[code_token_length] = token;
  code_token_length += 1;
  std::pair<std::vector<int>, std::vector<uint16_t>> result;
  result.first = token;
  result.second = last_hidden;
  return result;
}

py::dict TTSLlama::generate_code(std::vector<int> &history_tokens, int idx_spk, std::vector<float> &spk_emb, int EOS, float temp_tempreature)
{
  printf("generate code\n");
  py::dict result;
  if (history_tokens.empty())
  {
    printf("Sorry: your text is empty!!\n");
    history_tokens.clear();
    return result;
  }

  // make sure token not too largace
  if ((int)history_tokens.size() > SEQLEN - 10)
  {
    history_tokens.clear();
    printf("Error: your text is too large!\n");
    return result;
  }

  temperature = temp_tempreature;
  printf("temperature: %f\n", temperature);
  new_code_token_length = 0;
  std::vector<std::vector<int>> result_tokens;
  std::vector<std::vector<float>> result_hiddens;
  std::pair<std::vector<int>, std::vector<uint16_t>> curr_res = forward_first_code(history_tokens, idx_spk, spk_emb);
  std::vector<int> token = curr_res.first;

  bool has_EOS = false;
  for(int i = 0; i < int(token.size()); i++){
    if(token[i] == EOS){
      has_EOS = true;
      break;
    }
  }
  while (!has_EOS && code_token_length < SEQLEN)
  {
    result_tokens.emplace_back(token);
    // curr_res.second from uint16_t to float
    std::vector<float> curr_hiddens;
    curr_hiddens.resize(hidden_size);
    for(int i = 0; i < int(curr_res.second.size()); i++){
      curr_hiddens[i] = half2float(curr_res.second[i]);
    }
    // debug
    if(DEBUGGING) {
      printf("\n================code_token_length = %d", code_token_length);
      for(int i = 0; i < hidden_size; i++){
        if(i%5==0){printf("\n");}
        printf("%.04f ", curr_hiddens[i]);
      }
      printf("\n");
    }
    result_hiddens.emplace_back(std::move(curr_hiddens));
    new_code_token_length += 1;

    curr_res = forward_next_code();
    token = curr_res.first;
    for(int i = 0; i < int(token.size()); i++){
      if(token[i] == EOS){
        has_EOS = true;
        break;
      }
    }
  }
  result["tokens"] = result_tokens;
  result["hiddens"] = result_hiddens;
  return result;
}

std::pair<std::vector<float>, std::vector<float>> TTSLlama::forward_first_code_core(std::vector<int> &tokens, int idx_spk, std::vector<float> spk_emb)
{
  printf("forward_first_code\n");
  std::vector<int> position_id(SEQLEN, 0);
  std::vector<uint16_t> attention_mask(SEQLEN * SEQLEN, ATTENTION_MASK);
  
  for(int i = 0; i < int(tokens.size()); i++) 
  {
    std::fill(visited_code_tokens[i].begin(), visited_code_tokens[i].end(), tokens[i]);
  }

  code_token_length = tokens.size();

  for (int i = 0; i < code_token_length; i++)
  {
    position_id[i] = i;
  }
  for (int i = 0; i < code_token_length; i++)
  {
    for (int j = 0; j < SEQLEN; j++)
    {
      if (j <= i)
      {
        attention_mask[i * SEQLEN + j] = 0;
      }
    }
  }

  // forward embeding
  // i == 0, use text_emb
  tokens.resize(SEQLEN, 0);
  auto &in_mem = net_embed_text->stages[0].input_mems[0]; 
  auto &out_mem = net_embed_text->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)tokens.data()); // i == 0##########
  printf("token_length: %d\n", code_token_length);
  net_launch(net_embed_text); // prefil embedding

  if(DEBUGGING) {
    std::vector<uint16_t> tmp_emb(SEQLEN*hidden_size, 0); // debug
    bm_memcpy_d2s(bm_handle, tmp_emb.data(), out_mem); // debug
    // debug
    for(int t = 0; t < code_token_length; t++) {
      for (int i = 0; i < hidden_size; i++)
      {
        if(i%5 == 0) printf("\n");
        printf("%.04f ", half2float(tmp_emb[t*hidden_size + i]));
      }
      printf("================== \n");
    }
  }
  
  // the embedding of empty_spk --> spk_emb
  if (idx_spk != -1)
  {
    std::vector<uint16_t> tmp(hidden_size, 0);
    for(int i = 0; i < hidden_size; i++)
    {
      tmp[i] = float2uint16(spk_emb[i]);
    }
    int byte_size = out_mem.size / hidden_size;
    bm_memcpy_s2d_partial_offset(bm_handle, out_mem, (void *)tmp.data(), hidden_size * byte_size, idx_spk * hidden_size * byte_size); // bytesize, offset
  }

  // forward blocks
  for (int idx = 0; idx < NUM_LAYERS; idx++)
  {
    auto &in0_mem = net_blocks[idx]->stages[0].input_mems[0];
    auto &in1_mem = net_blocks[idx]->stages[0].input_mems[1];
    auto &in2_mem = net_blocks[idx]->stages[0].input_mems[2];
    d2d(in0_mem, out_mem);
    if (idx == 0)
    {
      // only first time need copy
      bm_memcpy_s2d(bm_handle, in1_mem, (void *)position_id.data());
      bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
    }
    net_launch(net_blocks[idx]);
    out_mem = net_blocks[idx]->stages[0].output_mems[0]; // hiddens
    
    d2d(past_key[idx], net_blocks[idx]->stages[0].output_mems[1]);
    d2d(past_value[idx], net_blocks[idx]->stages[0].output_mems[2]);

    // debug
    if(DEBUGGING) {
      std::vector<uint16_t> temp(hidden_size*SEQLEN, 0);
      bm_memcpy_d2s(bm_handle, temp.data(), out_mem);
      printf("\nforward first code layer %d\n", idx);
      for (int i = 0; i < hidden_size; i++)
      {
        if(i%5 == 0) printf("\n");
        printf("%.04f ", half2float(temp[(code_token_length-1)*hidden_size + i]));
      }
    }
  }
  // (bm_handle, past_value[idx], token_offset, out2_mem, 0, bytes);
  // forward lmhead
  int bytes = out_mem.size / SEQLEN;
  auto &lm_in_mem = net_lm_code->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm_code->stages[0].output_mems[0];
  bm_memcpy_d2d_byte(bm_handle, lm_in_mem, 0, out_mem,
                     (code_token_length - 1) * bytes, bytes);

  if(DEBUGGING) { 
    printf("\nBefore lmhead\n"); // bug here
    std::vector<uint16_t> temp0(hidden_size, 0);
    bm_memcpy_d2s(bm_handle, temp0.data(), lm_in_mem);
    for (int i = 0; i < int(temp0.size()); i++)
    {
      if(i%5 == 0) printf("\n");
      printf("%.04f ", half2float(temp0[i]));
    }
  }

  net_launch(net_lm_code); // B

  // debug
  if(DEBUGGING) { 
    std::vector<float> temp(626*4, 0);
    bm_memcpy_d2s(bm_handle, temp.data(), lm_out_mem);
    printf("\nAfter lmhead\n");
    for (int i = 0; i < int(temp.size()); i++)
    {
      if (i % 626 == 0) printf(" ====================== ");
      if(i%5 == 0) printf("\n");
      printf("%.04f ", temp[i]);
    }
  }

  std::vector<uint16_t> last_hidden(hidden_size, 0);
  bm_memcpy_d2s_partial_offset(bm_handle, (void *)last_hidden.data(), out_mem, bytes, (code_token_length - 1) * bytes); // ##### 1*512*12*64*[2B]
  if(DEBUGGING) {
    printf("\nlast hidden\n"); 
    for (int i = 0; i < int(last_hidden.size()); i++) 
    { printf("%.04f ", half2float(last_hidden[i])); }
    printf("\n");
  }

  std::vector<float> last_hidden_float(hidden_size, 0);
  for (int i = 0; i < int(last_hidden.size()); i++)
  {
    last_hidden_float[i] = half2float(last_hidden[i]);
  }

  std::pair<std::vector<float>, std::vector<float>> result;

  std::vector<float> temp(NUM_VQ*AUDIO_CODE_SIZE, 0); //1x626x4xf32
  bm_memcpy_d2s(bm_handle, temp.data(), lm_out_mem);
  printf("\nAfter lmhead\n");
  for (int i = 0; i < int(temp.size()); i++)
  {
    if (i % 626 == 0) printf(" ====================== ");
    if(i%5 == 0) printf("\n");
    printf("%.04f ", temp[i]);
  }

  result.first = std::move(temp);
  result.second = std::move(last_hidden_float);
  return result;
}

std::pair<std::vector<float>, std::vector<float>> TTSLlama::forward_next_code_core(std::vector<int> &newtoken)
{
  visited_code_tokens[code_token_length] = newtoken;
  code_token_length += 1;
  std::vector<int> cur_token = visited_code_tokens[code_token_length - 1];

  std::vector<uint16_t> attention_mask(SEQLEN + 1, 0);
  for (int i = code_token_length - 1; i < SEQLEN; i++)
  {
    attention_mask[i] = ATTENTION_MASK;
  }
  int32_t position_id = code_token_length - 1;

  // embedding
  auto &in_mem = net_embed_code_cache->stages[0].input_mems[0];
  auto &out_mem = net_embed_code_cache->stages[0].output_mems[0];
  bm_memcpy_s2d(bm_handle, in_mem, (void *)cur_token.data());
  net_launch(net_embed_code_cache);

  if (DEBUGGING) {
    std::vector<int> tmp(4);
    bm_memcpy_d2s(bm_handle, tmp.data(), in_mem);
    printf("\nnet_embed_code_cache in\n");
    for (int i = 0; i < 4; i++)
    {
      printf("%d  ", tmp[i]);
    }
    std::vector<uint16_t> emb_out(hidden_size);
    bm_memcpy_d2s(bm_handle, emb_out.data(), out_mem);
    printf("\nnet_embed_code_cache out\n");
    for (int i = 0; i < hidden_size; i++)
    {
      if(i%5 == 0) printf("\n");
      printf("%.04f ", half2float(emb_out[i]));
    }
  }
  // blocks
  int bytes =
      bm_mem_get_device_size(net_blocks_cache[0]->stages[0].output_mems[1]);
  int token_offset = (code_token_length - 1) * bytes;
  for (int idx = 0; idx < NUM_LAYERS; idx++)
  {
    auto &in0_mem = net_blocks_cache[idx]->stages[0].input_mems[0];
    auto &in1_mem = net_blocks_cache[idx]->stages[0].input_mems[1];
    auto &in2_mem = net_blocks_cache[idx]->stages[0].input_mems[2];
    auto &in3_mem = net_blocks_cache[idx]->stages[0].input_mems[3];
    auto &in4_mem = net_blocks_cache[idx]->stages[0].input_mems[4];
    auto &out0_mem = net_blocks_cache[idx]->stages[0].output_mems[0];
    auto &out1_mem = net_blocks_cache[idx]->stages[0].output_mems[1];
    auto &out2_mem = net_blocks_cache[idx]->stages[0].output_mems[2];
    d2d(in0_mem, out_mem);
    if (io_alone)
    {
      if (idx == 0)
      {
        bm_memcpy_s2d(bm_handle, in1_mem, (void *)&position_id);
        bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
      }
      else
      {
        d2d(in1_mem, net_blocks_cache[0]->stages[0].input_mems[1]);
        d2d(in2_mem, net_blocks_cache[0]->stages[0].input_mems[2]);
      }
    }
    else
    {
      if (idx == 0)
      {
        bm_memcpy_s2d(bm_handle, in1_mem, (void *)&position_id);
        bm_memcpy_s2d(bm_handle, in2_mem, (void *)attention_mask.data());
      }
      d2d(in3_mem, past_key[idx]);
      d2d(in4_mem, past_value[idx]);
    }
    net_launch(net_blocks_cache[idx]);
    out_mem = out0_mem;
    bm_memcpy_d2d_byte(bm_handle, past_key[idx], token_offset, out1_mem, 0,
                       bytes);
    bm_memcpy_d2d_byte(bm_handle, past_value[idx], token_offset, out2_mem, 0,
                       bytes);
  }

  std::vector<uint16_t> last_hidden(hidden_size, 0);  // debug
  bm_memcpy_d2s(bm_handle, (void *)last_hidden.data(), out_mem);

  std::vector<float> last_hidden_float(hidden_size, 0); //1x626x4xf32
  for (int i = 0; i < int(last_hidden.size()); i++)
  {
    last_hidden_float[i] = half2float(last_hidden[i]);
  }

  // forward lmhead
  auto &lm_in_mem = net_lm_code->stages[0].input_mems[0];
  auto &lm_out_mem = net_lm_code->stages[0].output_mems[0];
  d2d(lm_in_mem, out_mem);
  net_launch(net_lm_code);

  std::vector<float> temp(NUM_VQ*AUDIO_CODE_SIZE, 0); //1x626x4xf32
  bm_memcpy_d2s(bm_handle, temp.data(), lm_out_mem);

  std::pair<std::vector<float>, std::vector<float>> result;
  result.first = std::move(temp);
  result.second = std::move(last_hidden_float);
  return result;
}

PYBIND11_MODULE(llama, m)
{
  pybind11::class_<TTSLlama>(m, "TTSLlama")
      .def(pybind11::init<>())
      .def("init", &TTSLlama::init)
      .def("deinit", &TTSLlama::deinit)
      .def("forward_first_text", &TTSLlama::forward_first_text)
      .def("forward_next_text", &TTSLlama::forward_next_text)
      .def("generate_text", &TTSLlama::generate_text)
      .def("forward_first_code", &TTSLlama::forward_first_code)
      .def("forward_next_code", &TTSLlama::forward_next_code)
      .def("generate_code", &TTSLlama::generate_code)
      .def("forward_first_code_core", &TTSLlama::forward_first_code_core)
      .def("forward_next_code_core", &TTSLlama::forward_next_code_core)
      .def_readwrite("generation_mode_text", &TTSLlama::generation_mode_text)
      .def_readwrite("generation_mode_code", &TTSLlama::generation_mode_code)
      .def_readwrite("SEQLEN", &TTSLlama::SEQLEN) // read SEQLEN in pipeline.py
      .def_readwrite("temperature", &TTSLlama::temperature)
      .def_readwrite("top_p", &TTSLlama::top_p)
      .def_readwrite("repeat_penalty", &TTSLlama::repeat_penalty)
      .def_readwrite("repeat_last_n", &TTSLlama::repeat_last_n)
      .def_readwrite("max_new_tokens", &TTSLlama::max_new_tokens)
      .def_readwrite("DEBUGGING", &TTSLlama::DEBUGGING);
}