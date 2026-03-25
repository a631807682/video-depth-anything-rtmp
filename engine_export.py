import tensorrt as trt

def rebuild_engine(onnx_path, engine_path):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    config = builder.create_builder_config()
    
    # 1. 核心修复：配置动态维度 Profile
    profile = builder.create_optimization_profile()
    input_name = "pixel_values" # 确保这里和你之前查到的输入名一致
    # 设置 [Min, Opt, Max] 形状，这里我们全部固定为 518
    fixed_shape = (1, 3, 518, 518)
    profile.set_shape(input_name, fixed_shape, fixed_shape, fixed_shape)
    config.add_optimization_profile(profile)

    # 2. 4090 开启 FP16 加速
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    # 3. 解析 ONNX
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(f"解析错误: {parser.get_error(error)}")
            return

    print(f"正在构建 Engine (输入节点: {input_name})...")
    # 4. 构建并序列化
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("❌ 构建失败，请检查输入节点名称是否确实为 'pixel_values'")
        return

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"✅ Engine 已成功保存至: {engine_path}")

if __name__ == "__main__":
    # 确保使用你那个包含 pixel_values 节点的 ONNX
    rebuild_engine("./pretrained/depth_anything_v2_vits_fp16.onnx", "./pretrained/depth_vits_fp16.engine")
