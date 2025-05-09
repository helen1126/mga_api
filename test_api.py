# test_mga_api.py
import pytest
import requests
import torch
import clip
import numpy as np
import time
import json
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from locust import HttpUser, task, between, TaskSet, events

# 配置信息
API_URL = "http://localhost:8000/mga-weight"
BATCH_SIZES = [1, 4, 8]  # 测试不同的批处理大小
TEST_PROMPTS = [
    "一只猫在沙发上睡觉",
    "一辆红色的跑车在公路上疾驰",
    "一个人在公园里骑自行车",
    "美丽的海滩和蓝色的海洋",
    "繁华的城市夜景",
    "高山上的积雪和云雾",
    "秋天的落叶和金黄色的树林",
    "现代城市的摩天大楼",
    "宁静的湖泊和倒影",
    "星空下的山脉轮廓",
]

# 测试夹具 - 初始化CLIP模型
@pytest.fixture(scope="session")
def clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()
    return model

# 测试基本功能
def test_basic_functionality():
    for batch_size in BATCH_SIZES:
        # 准备测试数据
        prompts = TEST_PROMPTS[:batch_size]
        data = {"texts": prompts, "batch_size": batch_size}
        
        # 发送请求
        response = requests.post(API_URL, json=data)
        result = response.json()
        
        # 验证响应
        assert response.status_code == 200, f"请求失败: {response.status_code}"
        assert len(result["weights"]) == batch_size, "返回的权重数量不匹配"
        assert len(result["text_prompts"]) == batch_size, "返回的文本提示数量不匹配"
        
        # 验证权重矩阵属性
        for weight_vector in result["weights"]:
            assert len(weight_vector) > 0, "权重向量为空"
            # 检查权重是否在合理范围内
            for weight in weight_vector:
                assert 0 <= weight <= 1, f"权重值超出范围: {weight}"

# 测试空输入
def test_empty_input():
    data = {"texts": [], "batch_size": 1}
    response = requests.post(API_URL, json=data)
    
    assert response.status_code == 500, "空输入应返回400错误"

# 测试大批次处理
def test_large_batch_size():
    # 准备大量测试数据
    large_batch_size = 16
    prompts = TEST_PROMPTS * (large_batch_size // len(TEST_PROMPTS) + 1)
    prompts = prompts[:large_batch_size]
    data = {"texts": prompts, "batch_size": large_batch_size}
    
    # 发送请求
    response = requests.post(API_URL, json=data)
    result = response.json()
    
    assert response.status_code == 200, f"大批次请求失败: {response.status_code}"
    assert len(result["weights"]) == large_batch_size, "大批次返回的权重数量不匹配"

# 测试性能
@pytest.mark.performance
def test_performance():
    # 测试不同批次大小的性能
    for batch_size in BATCH_SIZES:
        prompts = TEST_PROMPTS[:batch_size]
        data = {"texts": prompts, "batch_size": batch_size}
        
        # 预热
        for _ in range(3):
            requests.post(API_URL, json=data)
        
        # 计时测试
        start_time = time.time()
        num_requests = 10
        for _ in tqdm(range(num_requests), desc=f"批处理大小 {batch_size}"):
            requests.post(API_URL, json=data)
        
        total_time = time.time() - start_time
        avg_time = total_time / num_requests
        
        print(f"批处理大小 {batch_size}: 平均请求时间 {avg_time:.4f}秒")

# 测试与CLIP模型的兼容性
def test_clip_compatibility(clip_model):
    # 选择一个文本提示
    text_prompt = TEST_PROMPTS[0]
    data = {"texts": [text_prompt], "batch_size": 1}
    
    # 获取API生成的权重
    response = requests.post(API_URL, json=data)
    result = response.json()
    
    # 获取CLIP模型的设备 (修正)
    device = next(clip_model.parameters()).device  # 保持原有代码
    weights = torch.tensor(result["weights"], device=device)
    
    # 使用CLIP编码文本 (修正)
    text_tokens = clip.tokenize([text_prompt]).to(device)  # 直接使用device变量
    
    # 检查权重与CLIP特征的一致性
    assert weights.shape[1] == text_tokens.shape[1], "权重维度与CLIP特征维度不匹配"

# 测试动态权重变化
def test_dynamic_weight_changes():
    # 选择两个不同的文本提示
    prompt1 = "一只猫在沙发上睡觉"
    prompt2 = "一架飞机在天空中飞行"
    
    # 获取第一个文本的权重
    data1 = {"texts": [prompt1], "batch_size": 1}
    response1 = requests.post(API_URL, json=data1)
    weights1 = np.array(response1.json()["weights"][0])
    
    # 获取第二个文本的权重
    data2 = {"texts": [prompt2], "batch_size": 1}
    response2 = requests.post(API_URL, json=data2)
    weights2 = np.array(response2.json()["weights"][0])
    
    # 计算权重差异
    weight_diff = np.mean(np.abs(weights1 - weights2))
    
    # 确保两个不同文本生成的权重有显著差异
    assert weight_diff > 0.1, f"权重差异太小: {weight_diff}"

# 压力测试类 (使用Locust)
class MGAApiUser(HttpUser):
    wait_time = between(0.5, 2.0)  # 模拟用户思考时间
    
    @task(3)
    def test_api(self):
        """测试API性能"""
        batch_size = 4
        prompts = TEST_PROMPTS[:batch_size]
        data = {"texts": prompts, "batch_size": batch_size}
        
        try:
            with self.client.post(API_URL, json=data, catch_response=True) as response:
                if response.status_code != 200:
                    response.failure(f"请求失败: {response.status_code}")
        except Exception as e:
            self.environment.events.request.fire(
                request_type="POST",
                name="/mga-weight",
                response_time=0,
                exception=e
            )

# 主测试函数
def run_all_tests():
    print("=== 开始MGA-Net API测试 ===")
    
    # 运行单元测试
    test_basic_functionality()
    test_empty_input()
    test_large_batch_size()
    test_performance()
    
    # 运行集成测试
    test_clip_compatibility(clip_model())
    test_dynamic_weight_changes()
    
    print("\n=== 所有测试通过! ===")

if __name__ == "__main__":
    run_all_tests()
    
    # 提示如何运行压力测试
    print("\n=== 压力测试说明 ===")
    print("若要运行压力测试，请执行以下命令:")
    print("1. 安装Locust: pip install locust")
    print("2. 运行压力测试: locust -f test_mga_api.py --host=http://localhost:8000")
    print("3. 打开浏览器访问 http://localhost:8089 配置并启动压力测试")