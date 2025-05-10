# stress_test_mganet.py
from locust import HttpUser, task, between, TaskSet
import torch
import clip
import torch.nn.functional as F
import numpy as np
import random
import os
from typing import List, Dict, Any

# 模型路径配置
CHECKPOINT_PATH = "checkpoint_v6.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模拟模型定义 (与原项目保持一致)
class MGAEmbedding(torch.nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=512):
        super().__init__()
        # 模拟加载预训练权重
        if not os.path.exists(CHECKPOINT_PATH):
            pretrained = torch.randn(vocab_size, embed_dim) * 0.02
            torch.save({"embed_weights": pretrained}, CHECKPOINT_PATH)
        state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        self.base_embed = torch.nn.Embedding.from_pretrained(
            state_dict["embed_weights"], freeze=False
        )
        self.projection = torch.nn.Sequential(
            torch.nn.Conv1d(embed_dim, embed_dim, 3, groups=embed_dim, padding=1),
            torch.nn.GELU()
        )
        self.layer_norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, input_ids):
        embeddings = self.base_embed(input_ids)
        projected = self.projection(embeddings.permute(0, 2, 1))
        projected = projected.permute(0, 2, 1)
        normalized = self.layer_norm(projected)
        return F.normalize(normalized, p=2, dim=-1)

class DynamicWeightMatrix(torch.nn.Module):
    def __init__(self, num_attributes=512, history_decay=0.9):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.randn(num_attributes, 1) * 0.01)
        self.stability = torch.nn.Parameter(torch.ones(num_attributes, 1))
        self.temperature = torch.nn.Parameter(torch.ones(1))
        self.register_buffer(
            "historical_weights",
            torch.ones(1, num_attributes) / num_attributes
        )
        self.history_decay = history_decay
        self.epsilon = 1e-8

    def forward(self, x, epoch=None):
        batch_size = x.size(0)
        adapted_x = x[:, :self.alpha.shape[0]].unsqueeze(-1)
        scaled_logits = (self.alpha * adapted_x) / self.temperature.clamp(min=0.1, max=100)
        
        # 数值稳定的softmax
        max_logits = scaled_logits.max(dim=1, keepdim=True).values.detach()
        stable_logits = scaled_logits - max_logits
        raw_weights = F.softmax(stable_logits + self.epsilon, dim=1)
        
        # 更新历史权重
        if self.training:
            current_weights = raw_weights.squeeze(-1).mean(dim=0).detach().view(1, -1)
            self.historical_weights = (
                self.history_decay * self.historical_weights +
                (1 - self.history_decay) * current_weights
            )
        
        # 应用稳定性和历史权重
        stability = self.stability.t().expand(batch_size, -1)
        historical = self.historical_weights.expand(batch_size, -1).to(x.device)
        weighted_weights = stability * raw_weights.squeeze(-1) + historical
        return F.softmax(weighted_weights, dim=1)

class MGANetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = MGAEmbedding()
        self.weight_matrix = DynamicWeightMatrix()
        self.clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, text_prompts=None):
        embeddings = self.embedding(input_ids)
        clip_features = None
        
        if text_prompts is not None:
            text_tokens = clip.tokenize(text_prompts).to(DEVICE)
            with torch.no_grad():
                clip_features = self.clip_model.encode_text(text_tokens).float()
        
        # 计算权重并应用
        weights = self.weight_matrix(embeddings.mean(dim=1))
        weighted_features = torch.einsum('bsd,bd->bsd', embeddings, weights)
        return weighted_features

# 压力测试任务集
# 优化建议1：使用类变量共享模型实例
class MGATaskSet(TaskSet):
    _model = None  # 类变量共享模型
    
    @classmethod
    def get_model(cls):
        if cls._model is None:
            cls._model = MGANetwork().to(DEVICE)
            cls._model.eval()
        return cls._model

    def on_start(self):
        self.model = self.get_model() 
        self.vocab_size = 1000
        self.max_memory = 0  
        
        # 增加更多多样化的测试文本提示
        self.text_prompts = [
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
            "热带雨林中的野生动物",
            "沙漠中的日落和沙丘",
            "古老城堡的石墙和塔楼",
            "繁忙的市场和人群",
            "宁静的乡村和田野",
            "冰雪覆盖的极地景观"
        ]
        print("测试环境初始化完成")
    
    @task(3)  # 权重为3，表示执行频率是其他任务的3倍
    def test_embedding_layer(self):
        """测试嵌入层性能"""
        batch_size = random.randint(8, 32)
        seq_length = random.randint(10, 50)
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_length)).to(DEVICE)
        
        try:
            with torch.no_grad():
                output = self.model.embedding(input_ids)
            
            # 记录测试结果 - 使用新的request事件
            self.user.environment.events.request.fire(
                request_type="EMBEDDING",
                name="MGAEmbedding",
                response_time=0,  # 由于直接调用，时间统计由Locust自动完成
                response_length=output.numel(),
                exception=None
            )
        except Exception as e:
            # 记录失败信息
            self.user.environment.events.request.fire(
                request_type="EMBEDDING",
                name="MGAEmbedding",
                response_time=0,
                response_length=0,
                exception=e
            )
    
    @task(2)
    def test_dynamic_weights(self):
        """测试动态权重层性能"""
        batch_size = random.randint(8, 32)
        feature_dim = 512
        input_features = torch.randn(batch_size, feature_dim).to(DEVICE)
        
        try:
            with torch.no_grad():
                weights = self.model.weight_matrix(input_features)
            
            # 记录测试结果 - 使用新的request事件
            self.user.environment.events.request.fire(
                request_type="WEIGHTS",
                name="DynamicWeightMatrix",
                response_time=0,
                response_length=weights.numel(),
                exception=None
            )
        except Exception as e:
            # 记录失败信息
            self.user.environment.events.request.fire(
                request_type="WEIGHTS",
                name="DynamicWeightMatrix",
                response_time=0,
                response_length=0,
                exception=e
            )
    
    @task(1)
    def test_full_model(self):
        try:
            batch_size = random.randint(8, 32)
            seq_length = random.randint(10, 50)
            input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_length)).to(DEVICE)
            prompts = random.sample(self.text_prompts, min(batch_size, len(self.text_prompts)))  # Add min() check
            
            with torch.cuda.amp.autocast():
                output = self.model(input_ids, prompts)
            
            current_mem = torch.cuda.memory_allocated()
            self.max_memory = max(self.max_memory, current_mem)
            
            self.user.environment.events.request.fire(
                request_type="FULL",
                name="MGANetwork",
                response_time=0,
                response_length=output.numel(),
                exception=None,
                memory_usage=current_mem,
                peak_memory=self.max_memory
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("显存不足，已清理缓存")
        except Exception as e:
            print(f"CUDA错误: {str(e)}")
            print(torch.cuda.memory_summary())
            self.user.environment.events.request.fire(
                request_type="FULL",
                name="MGANetwork",
                response_time=0,
                response_length=0,
                exception=e
            )

# 用户类定义
class MGAUser(HttpUser):
    tasks = [MGATaskSet]
    min_wait = 500
    max_wait = 2000
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 限制CUDA线程数
        torch.set_num_threads(4)  # 根据CPU核心数调整
        # 确保CLIP模型已加载
        try:
            clip.load("ViT-B/32", device=DEVICE)
        except Exception as e:
            print(f"加载CLIP模型时出错: {e}")
            print("请确保已安装CLIP: pip install git+https://github.com/openai/CLIP.git")

if __name__ == "__main__":
    print(DEVICE)
    print(torch.cuda.is_available())