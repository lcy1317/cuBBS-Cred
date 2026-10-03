#include "cubbs_bbs_gpu_abi.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

extern "C" int32_t cubbs_bbs_gpu_prepare_b_batch_v2(
    uint32_t abi_version, CuBbsBbsGpuPrepareBItem *items, size_t item_count,
    CuBbsBbsGpuResult *results);

extern "C" int32_t cubbs_bbs_gpu_g1_msm_batch_v2(
    uint32_t abi_version, CuBbsBbsGpuG1MsmItem *items, size_t item_count,
    CuBbsBbsGpuResult *results);

extern "C" int32_t cubbs_bbs_gpu_g1_msm_common_points_batch_v2(
    uint32_t abi_version, CuBbsBbsGpuG1MsmCommonPointsBatchItem *items,
    size_t group_count, CuBbsBbsGpuResult *results);

extern "C" int32_t cubbs_bbs_gpu_g1_msm_fused_common_points_batch_v2(
    uint32_t abi_version, CuBbsBbsGpuG1MsmCommonPointsBatchItem *items,
    size_t group_count, CuBbsBbsGpuResult *results);

extern "C" int32_t cubbs_bbs_gpu_g1_add_batch_v2(
    uint32_t abi_version, CuBbsBbsGpuG1AddItem *items, size_t item_count,
    CuBbsBbsGpuResult *results);

static std::vector<uint8_t> hex_to_bytes(const std::string &hex) {
  std::vector<uint8_t> out;
  out.reserve(hex.size() / 2);
  for (size_t i = 0; i < hex.size(); i += 2) {
    out.push_back(static_cast<uint8_t>(std::stoul(hex.substr(i, 2), nullptr, 16)));
  }
  return out;
}

static std::string hex_bytes(const uint8_t *bytes, size_t len) {
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (size_t i = 0; i < len; ++i) {
    out << std::setw(2) << static_cast<unsigned>(bytes[i]);
  }
  return out.str();
}

static bool env_flag_enabled(const char *name) {
  const char *value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0 &&
         std::strcmp(value, "false") != 0 && std::strcmp(value, "FALSE") != 0 &&
         std::strcmp(value, "no") != 0 && std::strcmp(value, "NO") != 0;
}

int main() {
  const std::vector<uint8_t> public_key = hex_to_bytes(
      "b69063e3a4a48fded32888e595c648cea235b873ae7fc5357d3ab33b72eabdc6cb24628108e82b5155cfc1360be2563418a49c293be1622a76f7e97decb948b7458d532751148b4f5fc3119330847904769c048df6017b2f74b435d84d2fea88");
  const std::vector<uint8_t> header = hex_to_bytes(
      "4445464748494a4b4c4d4e4f505152535455565758595a5b5c5d5e5f606162636465666768696a6b6c6d6e6f707172737475767778797a7b7c7d7e7f80818283");
  const std::vector<std::vector<uint8_t>> scalars = {
      hex_to_bytes("54296a210d4bd61899213e657ed2487b63ed50a32177fdbc17834430b4dd148f"),
      hex_to_bytes("3f5ba1c9c0945a758be302933fa0acede3a982a3d67f28e35daca5764b841757"),
      hex_to_bytes("6f19a0cb6dcda81932152564c0a0b12a2efb9cf0cdf9e99fbba40fbe267eecde"),
      hex_to_bytes("60fec48b53956555245ebd88c21a2a47ad57cd2852c7a811333399bd2127569f"),
  };
  const std::string expected_b =
      "a094b89b5702dd07c1a0cf80d9d919241eb9cef724097ef340f9d6113f54372c2c82ba285cc1f6dc9a42717d69e4c7a9";
  const std::vector<std::vector<uint8_t>> msm_points = {
      hex_to_bytes("a8ce256102840821a3e94ea9025e4662b205762f9776b3a766c872b948f1fd225e7c59698588e70d11406d161b4e28c9"),
      hex_to_bytes("a9ec65b70a7fbe40c874c9eb041c2cb0a7af36ccec1bea48fa2ba4c2eb67ef7f9ecb17ed27d38d27cdeddff44c8137be"),
      hex_to_bytes("98cd5313283aaf5db1b3ba8611fe6070d19e605de4078c38df36019fbaad0bd28dd090fd24ed27f7f4d22d5ff5dea7d4"),
      hex_to_bytes("a31fbe20c5c135bcaa8d9fc4e4ac665cc6db0226f35e737507e803044093f37697a9d452490a970eea6f9ad6c3dcaa3a"),
      hex_to_bytes("b479263445f4d2108965a9086f9d1fdc8cde77d14a91c856769521ad3344754cc5ce90d9bc4c696dffbc9ef1d6ad1b62"),
      hex_to_bytes("ac0401766d2128d4791d922557c7b4d1ae9a9b508ce266575244a8d6f32110d7b0b7557b77604869633bb49afbe20035"),
  };
  const std::vector<std::vector<uint8_t>> msm_scalars = {
      hex_to_bytes("0000000000000000000000000000000000000000000000000000000000000001"),
      hex_to_bytes("20c08e554ea70cdbf1cfc538821e4707f9347e8bd0e9ace4b9cab7f77d174a25"),
      hex_to_bytes("54296a210d4bd61899213e657ed2487b63ed50a32177fdbc17834430b4dd148f"),
      hex_to_bytes("3f5ba1c9c0945a758be302933fa0acede3a982a3d67f28e35daca5764b841757"),
      hex_to_bytes("6f19a0cb6dcda81932152564c0a0b12a2efb9cf0cdf9e99fbba40fbe267eecde"),
      hex_to_bytes("60fec48b53956555245ebd88c21a2a47ad57cd2852c7a811333399bd2127569f"),
  };
  std::vector<CuBbsBbsGpuByteSlice> point_slices(msm_points.size());
  std::vector<CuBbsBbsGpuByteSlice> msm_scalar_slices(msm_scalars.size());
  std::array<uint8_t, 48> msm_out{};
  for (size_t i = 0; i < msm_points.size(); ++i) {
    point_slices[i] = CuBbsBbsGpuByteSlice{msm_points[i].data(), msm_points[i].size()};
    msm_scalar_slices[i] = CuBbsBbsGpuByteSlice{msm_scalars[i].data(), msm_scalars[i].size()};
  }

  CuBbsBbsGpuG1MsmItem msm_item{
      point_slices.data(),
      msm_scalar_slices.data(),
      point_slices.size(),
      CuBbsBbsGpuByteBuffer{msm_out.data(), msm_out.size(), 0},
  };
  CuBbsBbsGpuResult msm_result{};
  uint8_t expected_first_route = CUBBS_BBS_MSM_ROUTE_CUBBS_UNIVERSAL;
  const char *benchmark_route_env =
      std::getenv("CUBBS_BBS_MSM_BENCH_ROUTE");
  if (benchmark_route_env != nullptr) {
    const unsigned long requested = std::strtoul(benchmark_route_env, nullptr, 10);
    if (requested >= CUBBS_BBS_MSM_ROUTE_BLST_ITEM &&
        requested <= CUBBS_BBS_MSM_ROUTE_SPPARK_COMMON_POINTS_REFERENCE) {
      expected_first_route = static_cast<uint8_t>(requested);
    }
  }
  int32_t status =
      cubbs_bbs_gpu_g1_msm_batch_v2(CUBBS_BBS_GPU_ABI_VERSION, &msm_item, 1, &msm_result);
  if (status != CUBBS_BBS_GPU_SUCCESS || msm_result.code != CUBBS_BBS_GPU_SUCCESS ||
      msm_result.bytes_written != msm_out.size() ||
      msm_result.value != expected_first_route) {
    std::cerr << "g1_msm failed status=" << status << " code=" << msm_result.code
              << " bytes=" << msm_result.bytes_written
              << " route=" << static_cast<unsigned>(msm_result.value) << "\n";
    return 1;
  }
  const std::string actual_msm = hex_bytes(msm_out.data(), msm_out.size());
  if (actual_msm != expected_b) {
    std::cerr << "g1_msm mismatch\nexpected " << expected_b << "\nactual   " << actual_msm
              << "\n";
    return 1;
  }
  std::cout << "cuda g1_msm ABI smoke ok: route="
            << static_cast<unsigned>(msm_result.value)
            << " matches prepare-B vector\n";
  if (expected_first_route >= CUBBS_BBS_MSM_ROUTE_BLST_ITEM &&
      expected_first_route !=
          CUBBS_BBS_MSM_ROUTE_SPPARK_COMMON_POINTS_REFERENCE) {
    return 0;
  }

  const std::vector<uint8_t> generic_one_scalar = hex_to_bytes(
      "0000000000000000000000000000000000000000000000000000000000000001");
  std::vector<CuBbsBbsGpuByteSlice> mixed_points = point_slices;
  std::vector<CuBbsBbsGpuByteSlice> mixed_scalars = msm_scalar_slices;
  mixed_points.push_back(point_slices[1]);
  mixed_scalars.push_back(CuBbsBbsGpuByteSlice{
      generic_one_scalar.data(), generic_one_scalar.size()});
  std::array<uint8_t, 48> mixed_out0{};
  std::array<uint8_t, 48> mixed_out1{};
  std::array<CuBbsBbsGpuG1MsmItem, 2> mixed_items = {
      CuBbsBbsGpuG1MsmItem{
          mixed_points.data(), mixed_scalars.data(), point_slices.size(),
          CuBbsBbsGpuByteBuffer{mixed_out0.data(), mixed_out0.size(), 0}},
      CuBbsBbsGpuG1MsmItem{
          mixed_points.data() + point_slices.size(),
          mixed_scalars.data() + msm_scalar_slices.size(), 1u,
          CuBbsBbsGpuByteBuffer{mixed_out1.data(), mixed_out1.size(), 0}},
  };
  std::array<CuBbsBbsGpuResult, 2> mixed_results{};
  status = cubbs_bbs_gpu_g1_msm_batch_v2(
      CUBBS_BBS_GPU_ABI_VERSION, mixed_items.data(), mixed_items.size(),
      mixed_results.data());
  if (status != CUBBS_BBS_GPU_SUCCESS ||
      mixed_results[0].code != CUBBS_BBS_GPU_SUCCESS ||
      mixed_results[1].code != CUBBS_BBS_GPU_SUCCESS ||
      mixed_results[0].value != expected_first_route ||
      mixed_results[1].value != expected_first_route ||
      hex_bytes(mixed_out0.data(), mixed_out0.size()) != expected_b ||
      hex_bytes(mixed_out1.data(), mixed_out1.size()) !=
          hex_bytes(msm_points[1].data(), msm_points[1].size())) {
    std::cerr << "mixed-shape cuBBS-Cred generic route failed status=" << status
              << " route0=" << static_cast<unsigned>(mixed_results[0].value)
              << " route1=" << static_cast<unsigned>(mixed_results[1].value)
              << "\n";
    return 1;
  }
  std::cout << "cuda mixed-shape MSM smoke ok: all rows use selected route\n";

  const size_t universal_term_count = env_flag_enabled("CUBBS_BBS_GPU_ULTRA_SMOKE")
      ? 65538u
      : 4099u;
  const std::vector<uint8_t> zero_scalar(32u, 0u);
  std::vector<CuBbsBbsGpuByteSlice> universal_points(universal_term_count);
  std::vector<CuBbsBbsGpuByteSlice> universal_scalars(universal_term_count);
  for (size_t term = 0u; term < universal_term_count; ++term) {
    universal_points[term] = point_slices[term % point_slices.size()];
    universal_scalars[term] = term < msm_scalar_slices.size()
        ? msm_scalar_slices[term]
        : CuBbsBbsGpuByteSlice{zero_scalar.data(), zero_scalar.size()};
  }
  std::array<uint8_t, 48> universal_out{};
  CuBbsBbsGpuByteBuffer universal_out_buffer{
      universal_out.data(), universal_out.size(), 0u};
  CuBbsBbsGpuG1MsmCommonPointsBatchItem universal_item{
      universal_points.data(), universal_points.size(),
      universal_scalars.data(), nullptr, 0u, 1u, &universal_out_buffer};
  CuBbsBbsGpuResult universal_result{};
  status = cubbs_bbs_gpu_g1_msm_common_points_batch_v2(
      CUBBS_BBS_GPU_ABI_VERSION, &universal_item, 1u, &universal_result);
  if (status != CUBBS_BBS_GPU_SUCCESS ||
      universal_result.code != CUBBS_BBS_GPU_SUCCESS ||
      universal_result.value != expected_first_route ||
      universal_result.bytes_written != universal_out.size() ||
      hex_bytes(universal_out.data(), universal_out.size()) != expected_b) {
    std::cerr << "universal common-points route failed status=" << status
              << " code=" << universal_result.code
              << " route=" << static_cast<unsigned>(universal_result.value)
              << "\n";
    return 1;
  }
  std::cout << "cuda " << universal_term_count
            << "-term common-points smoke ok: selected route\n";

  const std::vector<uint8_t> two_scalar =
      hex_to_bytes("0000000000000000000000000000000000000000000000000000000000000002");
  std::vector<std::vector<uint8_t>> msm_scalars_row2 = msm_scalars;
  msm_scalars_row2[0] = two_scalar;
  std::vector<CuBbsBbsGpuByteSlice> common_scalar_rows;
  common_scalar_rows.reserve(msm_scalars.size() * 2);
  for (const auto &scalar : msm_scalars) {
    common_scalar_rows.push_back(CuBbsBbsGpuByteSlice{scalar.data(), scalar.size()});
  }
  for (const auto &scalar : msm_scalars_row2) {
    common_scalar_rows.push_back(CuBbsBbsGpuByteSlice{scalar.data(), scalar.size()});
  }
  std::array<uint8_t, 48> common_out0{};
  std::array<uint8_t, 48> common_out1{};
  std::array<CuBbsBbsGpuByteBuffer, 2> common_outs = {
      CuBbsBbsGpuByteBuffer{common_out0.data(), common_out0.size(), 0},
      CuBbsBbsGpuByteBuffer{common_out1.data(), common_out1.size(), 0},
  };
  CuBbsBbsGpuG1MsmCommonPointsBatchItem common_item{
      point_slices.data(),
      point_slices.size(),
      common_scalar_rows.data(),
      nullptr,
      0,
      common_outs.size(),
      common_outs.data(),
  };
  std::array<CuBbsBbsGpuResult, 2> common_results{};
  status = cubbs_bbs_gpu_g1_msm_common_points_batch_v2(
      CUBBS_BBS_GPU_ABI_VERSION, &common_item, 1, common_results.data());
  if (status != CUBBS_BBS_GPU_SUCCESS ||
      common_results[0].code != CUBBS_BBS_GPU_SUCCESS ||
      common_results[1].code != CUBBS_BBS_GPU_SUCCESS ||
      common_results[0].bytes_written != common_out0.size() ||
      common_results[1].bytes_written != common_out1.size()) {
    std::cerr << "g1_msm common-points failed status=" << status
              << " code0=" << common_results[0].code
              << " code1=" << common_results[1].code << "\n";
    return 1;
  }
  if (hex_bytes(common_out0.data(), common_out0.size()) != expected_b) {
    std::cerr << "g1_msm common-points row0 mismatch\nexpected " << expected_b
              << "\nactual   " << hex_bytes(common_out0.data(), common_out0.size()) << "\n";
    return 1;
  }
  std::vector<CuBbsBbsGpuByteSlice> row2_scalar_slices(msm_scalars_row2.size());
  for (size_t i = 0; i < msm_scalars_row2.size(); ++i) {
    row2_scalar_slices[i] =
        CuBbsBbsGpuByteSlice{msm_scalars_row2[i].data(), msm_scalars_row2[i].size()};
  }
  std::array<uint8_t, 48> row2_check_out{};
  CuBbsBbsGpuG1MsmItem row2_check_item{
      point_slices.data(),
      row2_scalar_slices.data(),
      point_slices.size(),
      CuBbsBbsGpuByteBuffer{row2_check_out.data(), row2_check_out.size(), 0},
  };
  CuBbsBbsGpuResult row2_check_result{};
  status = cubbs_bbs_gpu_g1_msm_batch_v2(
      CUBBS_BBS_GPU_ABI_VERSION, &row2_check_item, 1, &row2_check_result);
  if (status != CUBBS_BBS_GPU_SUCCESS ||
      row2_check_result.code != CUBBS_BBS_GPU_SUCCESS ||
      row2_check_result.bytes_written != row2_check_out.size()) {
    std::cerr << "g1_msm common-points row2 check failed status=" << status
              << " code=" << row2_check_result.code << "\n";
    return 1;
  }
  if (hex_bytes(common_out1.data(), common_out1.size()) !=
      hex_bytes(row2_check_out.data(), row2_check_out.size())) {
    std::cerr << "g1_msm common-points row1 mismatch\nexpected "
              << hex_bytes(row2_check_out.data(), row2_check_out.size())
              << "\nactual   " << hex_bytes(common_out1.data(), common_out1.size()) << "\n";
    return 1;
  }
  std::cout << "cuda g1_msm common-points ABI smoke ok: rows match regular MSM\n";

  std::vector<uint8_t> packed_common_scalars(msm_scalars.size() * 2 * 32u);
  for (size_t term = 0; term < msm_scalars.size(); ++term) {
    std::memcpy(packed_common_scalars.data() + (term * 32u), msm_scalars[term].data(), 32u);
    std::memcpy(
        packed_common_scalars.data() + ((msm_scalars.size() + term) * 32u),
        msm_scalars_row2[term].data(), 32u);
  }
  std::array<uint8_t, 48> packed_common_out0{};
  std::array<uint8_t, 48> packed_common_out1{};
  std::array<CuBbsBbsGpuByteBuffer, 2> packed_common_outs = {
      CuBbsBbsGpuByteBuffer{packed_common_out0.data(), packed_common_out0.size(), 0},
      CuBbsBbsGpuByteBuffer{packed_common_out1.data(), packed_common_out1.size(), 0},
  };
  CuBbsBbsGpuG1MsmCommonPointsBatchItem packed_common_item{
      point_slices.data(),
      point_slices.size(),
      nullptr,
      packed_common_scalars.data(),
      point_slices.size() * 32u,
      packed_common_outs.size(),
      packed_common_outs.data(),
  };
  std::array<CuBbsBbsGpuResult, 2> packed_common_results{};
  status = cubbs_bbs_gpu_g1_msm_common_points_batch_v2(
      CUBBS_BBS_GPU_ABI_VERSION, &packed_common_item, 1,
      packed_common_results.data());
  if (status != CUBBS_BBS_GPU_SUCCESS ||
      packed_common_results[0].code != CUBBS_BBS_GPU_SUCCESS ||
      packed_common_results[1].code != CUBBS_BBS_GPU_SUCCESS ||
      packed_common_results[0].bytes_written != packed_common_out0.size() ||
      packed_common_results[1].bytes_written != packed_common_out1.size()) {
    std::cerr << "g1_msm packed common-points failed status=" << status
              << " code0=" << packed_common_results[0].code
              << " code1=" << packed_common_results[1].code << "\n";
    return 1;
  }
  if (hex_bytes(packed_common_out0.data(), packed_common_out0.size()) !=
          expected_b ||
      hex_bytes(packed_common_out1.data(), packed_common_out1.size()) !=
          hex_bytes(row2_check_out.data(), row2_check_out.size())) {
    std::cerr << "g1_msm packed common-points mismatch\nrow0 "
              << hex_bytes(packed_common_out0.data(), packed_common_out0.size())
              << "\nrow1   "
              << hex_bytes(packed_common_out1.data(), packed_common_out1.size())
              << "\n";
    return 1;
  }
  std::cout
      << "cuda g1_msm packed common-points ABI smoke ok: packed rows match slices\n";

  std::array<uint8_t, 48> fused_out0{};
  std::array<uint8_t, 48> fused_out1{};
  std::array<CuBbsBbsGpuByteBuffer, 2> fused_outs = {
      CuBbsBbsGpuByteBuffer{fused_out0.data(), fused_out0.size(), 0},
      CuBbsBbsGpuByteBuffer{fused_out1.data(), fused_out1.size(), 0},
  };
  CuBbsBbsGpuG1MsmCommonPointsBatchItem fused_item{
      point_slices.data(),
      point_slices.size(),
      common_scalar_rows.data(),
      nullptr,
      0,
      fused_outs.size(),
      fused_outs.data(),
  };
  std::array<CuBbsBbsGpuResult, 2> fused_results{};
  status = cubbs_bbs_gpu_g1_msm_fused_common_points_batch_v2(
      CUBBS_BBS_GPU_ABI_VERSION, &fused_item, 1, fused_results.data());
  const bool expect_naive_fused =
      env_flag_enabled("CUBBS_BBS_GPU_FUSED_COMMON_POINTS_NAIVE");
  const char *fused_label =
      expect_naive_fused ? "naive" : "production";
  // Both the production and the naive fused bodies must produce route code 2
  // and byte-for-byte match the regular common-points MSM output. The
  // "body pending" sentinel is no longer the default.
  if (status != CUBBS_BBS_GPU_SUCCESS ||
      fused_results[0].code != CUBBS_BBS_GPU_SUCCESS ||
      fused_results[1].code != CUBBS_BBS_GPU_SUCCESS ||
      fused_results[0].value != 2 || fused_results[1].value != 2 ||
      fused_results[0].bytes_written != fused_out0.size() ||
      fused_results[1].bytes_written != fused_out1.size() ||
      hex_bytes(fused_out0.data(), fused_out0.size()) != expected_b ||
      hex_bytes(fused_out1.data(), fused_out1.size()) !=
          hex_bytes(row2_check_out.data(), row2_check_out.size())) {
    std::cerr << "g1_msm fused common-points " << fused_label
              << " mismatch status=" << status
              << " code0=" << fused_results[0].code
              << " code1=" << fused_results[1].code
              << " route0=" << static_cast<unsigned>(fused_results[0].value)
              << " route1=" << static_cast<unsigned>(fused_results[1].value)
              << "\n";
    return 1;
  }
  std::cout << "cuda g1_msm fused common-points " << fused_label
            << " smoke ok: rows match regular MSM, route=2\n";

  // Repartition the exact cached byte stream as 47/49-byte slices. Without
  // pre-key validation this aliases the valid cache entry above and bypasses
  // point decoding even though neither slice is a valid compressed G1 point.
  std::vector<uint8_t> repartitioned_point_bytes;
  repartitioned_point_bytes.reserve(point_slices.size() * 48u);
  for (const auto &point : point_slices) {
    repartitioned_point_bytes.insert(
        repartitioned_point_bytes.end(), point.data, point.data + point.len);
  }
  auto malformed_point_slices = point_slices;
  malformed_point_slices[0] = CuBbsBbsGpuByteSlice{
      repartitioned_point_bytes.data(), 47u};
  malformed_point_slices[1] = CuBbsBbsGpuByteSlice{
      repartitioned_point_bytes.data() + 47u, 49u};
  for (size_t point = 2u; point < malformed_point_slices.size(); ++point) {
    malformed_point_slices[point] = CuBbsBbsGpuByteSlice{
        repartitioned_point_bytes.data() + (point * 48u), 48u};
  }
  fused_item.points = malformed_point_slices.data();
  std::array<CuBbsBbsGpuResult, 2> malformed_point_results{};
  status = cubbs_bbs_gpu_g1_msm_fused_common_points_batch_v2(
      CUBBS_BBS_GPU_ABI_VERSION, &fused_item, 1,
      malformed_point_results.data());
  const bool malformed_rejected =
      status == CUBBS_BBS_GPU_ERR_INVALID_INPUT ||
      (status == CUBBS_BBS_GPU_SUCCESS &&
       malformed_point_results[0].code == CUBBS_BBS_GPU_ERR_INVALID_INPUT);
  if (!malformed_rejected) {
    std::cerr << "fused malformed point cache-key validation failed status="
              << status << " code=" << malformed_point_results[0].code << "\n";
    return 1;
  }
  fused_item.points = point_slices.data();
  std::cout << "cuda fused malformed point cache-key validation ok\n";

  std::array<uint8_t, 48> fused_packed_out0{};
  std::array<uint8_t, 48> fused_packed_out1{};
  std::array<CuBbsBbsGpuByteBuffer, 2> fused_packed_outs = {
      CuBbsBbsGpuByteBuffer{fused_packed_out0.data(), fused_packed_out0.size(), 0},
      CuBbsBbsGpuByteBuffer{fused_packed_out1.data(), fused_packed_out1.size(), 0},
  };
  CuBbsBbsGpuG1MsmCommonPointsBatchItem fused_packed_item{
      point_slices.data(),
      point_slices.size(),
      nullptr,
      packed_common_scalars.data(),
      point_slices.size() * 32u,
      fused_packed_outs.size(),
      fused_packed_outs.data(),
  };
  std::array<CuBbsBbsGpuResult, 2> fused_packed_results{};
  status = cubbs_bbs_gpu_g1_msm_fused_common_points_batch_v2(
      CUBBS_BBS_GPU_ABI_VERSION, &fused_packed_item, 1,
      fused_packed_results.data());
  if (status != CUBBS_BBS_GPU_SUCCESS ||
      fused_packed_results[0].code != CUBBS_BBS_GPU_SUCCESS ||
      fused_packed_results[1].code != CUBBS_BBS_GPU_SUCCESS ||
      fused_packed_results[0].value != 2 || fused_packed_results[1].value != 2 ||
      fused_packed_results[0].bytes_written != fused_packed_out0.size() ||
      fused_packed_results[1].bytes_written != fused_packed_out1.size() ||
      hex_bytes(fused_packed_out0.data(), fused_packed_out0.size()) !=
          expected_b ||
      hex_bytes(fused_packed_out1.data(), fused_packed_out1.size()) !=
          hex_bytes(row2_check_out.data(), row2_check_out.size())) {
    std::cerr << "g1_msm fused packed common-points " << fused_label
              << " mismatch status=" << status
              << " code0=" << fused_packed_results[0].code
              << " code1=" << fused_packed_results[1].code
              << " route0=" << static_cast<unsigned>(fused_packed_results[0].value)
              << " route1=" << static_cast<unsigned>(fused_packed_results[1].value)
              << "\n";
    return 1;
  }
  std::cout << "cuda g1_msm fused packed common-points " << fused_label
            << " smoke ok: packed rows match slices, route=2\n";

  // Extended packed rows prepend one multiplier. Exercise a non-trivial
  // multiplier instead of scale=1 so byte order and Montgomery conversion are
  // covered: 2 * (1 * P0) must match a regular MSM row containing 2 * P0.
  std::vector<uint8_t> scaled_packed_row((point_slices.size() + 1u) * 32u, 0u);
  std::memcpy(scaled_packed_row.data(), two_scalar.data(), 32u);
  std::memcpy(scaled_packed_row.data() + 32u, generic_one_scalar.data(), 32u);
  std::array<uint8_t, 48> scaled_packed_out{};
  CuBbsBbsGpuByteBuffer scaled_packed_out_buffer{
      scaled_packed_out.data(), scaled_packed_out.size(), 0u};
  CuBbsBbsGpuG1MsmCommonPointsBatchItem scaled_packed_item{
      point_slices.data(), point_slices.size(), nullptr,
      scaled_packed_row.data(), (point_slices.size() + 1u) * 32u, 1u,
      &scaled_packed_out_buffer};
  CuBbsBbsGpuResult scaled_packed_result{};
  status = cubbs_bbs_gpu_g1_msm_fused_common_points_batch_v2(
      CUBBS_BBS_GPU_ABI_VERSION, &scaled_packed_item, 1u,
      &scaled_packed_result);

  std::vector<CuBbsBbsGpuByteSlice> scaled_expected_scalars(point_slices.size());
  scaled_expected_scalars[0] =
      CuBbsBbsGpuByteSlice{two_scalar.data(), two_scalar.size()};
  for (size_t term = 1u; term < scaled_expected_scalars.size(); ++term) {
    scaled_expected_scalars[term] =
        CuBbsBbsGpuByteSlice{zero_scalar.data(), zero_scalar.size()};
  }
  std::array<uint8_t, 48> scaled_expected_out{};
  CuBbsBbsGpuG1MsmItem scaled_expected_item{
      point_slices.data(), scaled_expected_scalars.data(), point_slices.size(),
      CuBbsBbsGpuByteBuffer{
          scaled_expected_out.data(), scaled_expected_out.size(), 0u}};
  CuBbsBbsGpuResult scaled_expected_result{};
  const int32_t scaled_expected_status = cubbs_bbs_gpu_g1_msm_batch_v2(
      CUBBS_BBS_GPU_ABI_VERSION, &scaled_expected_item, 1u,
      &scaled_expected_result);
  if (status != CUBBS_BBS_GPU_SUCCESS ||
      scaled_packed_result.code != CUBBS_BBS_GPU_SUCCESS ||
      scaled_packed_result.value != 2u ||
      scaled_packed_result.bytes_written != scaled_packed_out.size() ||
      scaled_expected_status != CUBBS_BBS_GPU_SUCCESS ||
      scaled_expected_result.code != CUBBS_BBS_GPU_SUCCESS ||
      hex_bytes(scaled_packed_out.data(), scaled_packed_out.size()) !=
          hex_bytes(scaled_expected_out.data(), scaled_expected_out.size())) {
    std::cerr << "g1_msm fused device row-scale mismatch status=" << status
              << " code=" << scaled_packed_result.code
              << " route=" << static_cast<unsigned>(scaled_packed_result.value)
              << " expected_status=" << scaled_expected_status
              << " expected_code=" << scaled_expected_result.code << "\n";
    return 1;
  }
  std::cout << "cuda g1_msm fused device row-scale smoke ok: "
               "scale=2 matches regular MSM\n";

  // The same packed layout must survive a forced universal fallback.
  const char *prior_route = std::getenv("CUBBS_BBS_MSM_BENCH_ROUTE");
  const bool had_prior_route = prior_route != nullptr;
  const std::string saved_route = had_prior_route ? prior_route : "";
  setenv("CUBBS_BBS_MSM_BENCH_ROUTE", "4", 1);
  scaled_packed_result = {};
  // The benchmark override belongs to the route-selecting ABI, not the
  // explicitly fused entry point (which deliberately selects its own path).
  status = cubbs_bbs_gpu_g1_msm_common_points_batch_v2(
      CUBBS_BBS_GPU_ABI_VERSION, &scaled_packed_item, 1u, &scaled_packed_result);
  if (had_prior_route) setenv("CUBBS_BBS_MSM_BENCH_ROUTE", saved_route.c_str(), 1);
  else unsetenv("CUBBS_BBS_MSM_BENCH_ROUTE");
  if (status != CUBBS_BBS_GPU_SUCCESS || scaled_packed_result.code != CUBBS_BBS_GPU_SUCCESS ||
      scaled_packed_result.value != 4u || scaled_packed_out != scaled_expected_out) {
    std::cerr << "universal row-scale regression failed: status=" << status
              << " code=" << scaled_packed_result.code
              << " route=" << static_cast<unsigned>(scaled_packed_result.value) << "\n";
    return 1;
  }
  std::cout << "cuda universal row-scale regression ok\n";

  const auto invalid_fr = hex_to_bytes(
      "73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001");
  auto invalid_slice_rows = common_scalar_rows;
  invalid_slice_rows[0] =
      CuBbsBbsGpuByteSlice{invalid_fr.data(), invalid_fr.size()};
  fused_item.scalar_rows = invalid_slice_rows.data();
  std::array<CuBbsBbsGpuResult, 2> invalid_slice_results{};
  status = cubbs_bbs_gpu_g1_msm_fused_common_points_batch_v2(
      CUBBS_BBS_GPU_ABI_VERSION, &fused_item, 1,
      invalid_slice_results.data());
  if (status != CUBBS_BBS_GPU_SUCCESS ||
      invalid_slice_results[0].code != CUBBS_BBS_GPU_ERR_INVALID_INPUT) {
    std::cerr << "fused scalar Fr-boundary slice validation failed status="
              << status << " code=" << invalid_slice_results[0].code << "\n";
    return 1;
  }

  auto invalid_packed_scalars = packed_common_scalars;
  std::copy(invalid_fr.begin(), invalid_fr.end(),
            invalid_packed_scalars.begin());
  fused_packed_item.scalar_rows_be = invalid_packed_scalars.data();
  std::array<CuBbsBbsGpuResult, 2> invalid_packed_results{};
  status = cubbs_bbs_gpu_g1_msm_fused_common_points_batch_v2(
      CUBBS_BBS_GPU_ABI_VERSION, &fused_packed_item, 1,
      invalid_packed_results.data());
  if (status != CUBBS_BBS_GPU_SUCCESS ||
      invalid_packed_results[0].code != CUBBS_BBS_GPU_ERR_INVALID_INPUT) {
    std::cerr << "fused scalar Fr-boundary packed validation failed status="
              << status << " code=" << invalid_packed_results[0].code << "\n";
    return 1;
  }
  std::cout << "cuda fused scalar Fr-boundary validation ok: slice and packed rejected\n";

  const std::vector<uint8_t> one_scalar =
      hex_to_bytes("0000000000000000000000000000000000000000000000000000000000000001");
  std::array<uint8_t, 48> add_out{};
  CuBbsBbsGpuG1AddItem add_item{
      CuBbsBbsGpuByteSlice{msm_out.data(), msm_out.size()},
      CuBbsBbsGpuByteSlice{msm_points[0].data(), msm_points[0].size()},
      CuBbsBbsGpuByteBuffer{add_out.data(), add_out.size(), 0},
  };
  CuBbsBbsGpuResult add_result{};
  status =
      cubbs_bbs_gpu_g1_add_batch_v2(CUBBS_BBS_GPU_ABI_VERSION, &add_item, 1, &add_result);
  if (status != CUBBS_BBS_GPU_SUCCESS || add_result.code != CUBBS_BBS_GPU_SUCCESS ||
      add_result.bytes_written != add_out.size()) {
    std::cerr << "g1_add failed status=" << status << " code=" << add_result.code
              << " bytes=" << add_result.bytes_written << "\n";
    return 1;
  }

  std::vector<CuBbsBbsGpuByteSlice> add_check_points = {
      CuBbsBbsGpuByteSlice{msm_out.data(), msm_out.size()},
      CuBbsBbsGpuByteSlice{msm_points[0].data(), msm_points[0].size()},
  };
  std::vector<CuBbsBbsGpuByteSlice> add_check_scalars = {
      CuBbsBbsGpuByteSlice{one_scalar.data(), one_scalar.size()},
      CuBbsBbsGpuByteSlice{one_scalar.data(), one_scalar.size()},
  };
  std::array<uint8_t, 48> add_check_out{};
  CuBbsBbsGpuG1MsmItem add_check_item{
      add_check_points.data(),
      add_check_scalars.data(),
      add_check_points.size(),
      CuBbsBbsGpuByteBuffer{add_check_out.data(), add_check_out.size(), 0},
  };
  CuBbsBbsGpuResult add_check_result{};
  status = cubbs_bbs_gpu_g1_msm_batch_v2(
      CUBBS_BBS_GPU_ABI_VERSION, &add_check_item, 1, &add_check_result);
  if (status != CUBBS_BBS_GPU_SUCCESS ||
      add_check_result.code != CUBBS_BBS_GPU_SUCCESS ||
      add_check_result.bytes_written != add_check_out.size()) {
    std::cerr << "g1_add check msm failed status=" << status
              << " code=" << add_check_result.code
              << " bytes=" << add_check_result.bytes_written << "\n";
    return 1;
  }
  if (hex_bytes(add_out.data(), add_out.size()) !=
      hex_bytes(add_check_out.data(), add_check_out.size())) {
    std::cerr << "g1_add mismatch\nexpected "
              << hex_bytes(add_check_out.data(), add_check_out.size()) << "\nactual   "
              << hex_bytes(add_out.data(), add_out.size()) << "\n";
    return 1;
  }
  std::cout << "cuda g1_add ABI smoke ok: native add matches two-term MSM\n";

  std::vector<CuBbsBbsGpuByteSlice> scalar_slices(scalars.size());
  std::array<uint8_t, 48> prepared_b{};

  for (size_t i = 0; i < scalars.size(); ++i) {
    scalar_slices[i] = CuBbsBbsGpuByteSlice{scalars[i].data(), scalars[i].size()};
  }

  CuBbsBbsGpuPrepareBItem item{
      CuBbsBbsGpuByteSlice{public_key.data(), public_key.size()},
      CuBbsBbsGpuByteSlice{header.data(), header.size()},
      scalar_slices.data(),
      scalar_slices.size(),
      CuBbsBbsGpuByteBuffer{prepared_b.data(), prepared_b.size(), 0},
  };
  CuBbsBbsGpuResult result{};

  status = cubbs_bbs_gpu_prepare_b_batch_v2(CUBBS_BBS_GPU_ABI_VERSION, &item, 1, &result);
  if (status != CUBBS_BBS_GPU_SUCCESS) {
    std::cerr << "prepare_b ABI status=" << status << "\n";
    return 1;
  }
  if (result.code == CUBBS_BBS_GPU_SUCCESS) {
    if (result.bytes_written != prepared_b.size()) {
      std::cerr << "prepare_b success wrote unexpected bytes=" << result.bytes_written << "\n";
      return 1;
    }
    const std::string actual_b = hex_bytes(prepared_b.data(), prepared_b.size());
    if (actual_b != expected_b) {
      std::cerr << "prepare_b mismatch\nexpected " << expected_b << "\nactual   " << actual_b
                << "\n";
      return 1;
    }
  } else if (result.code == CUBBS_BBS_GPU_ERR_INTERNAL && result.bytes_written == 0) {
    std::cout << "cuda prepare_b vector pending: MSM not implemented yet\n";
  } else {
    std::cerr << "prepare_b unexpected result code=" << result.code
              << " bytes=" << result.bytes_written << "\n";
    return 1;
  }

  scalar_slices[1].len = 31;
  status = cubbs_bbs_gpu_prepare_b_batch_v2(CUBBS_BBS_GPU_ABI_VERSION, &item, 1, &result);
  if (status != CUBBS_BBS_GPU_ERR_INVALID_INPUT) {
    std::cerr << "prepare_b invalid scalar length status=" << status << "\n";
    return 1;
  }

  std::cout << "cuda prepare_b ABI smoke ok: symbol exported, validation active, MSM pending\n";
  return 0;
}
