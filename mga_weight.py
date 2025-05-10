# api_service.py
import torch
import clip
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
from typing import List, Optional
import uvicorn

# 设置设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 导入MGA-Net相关模块
from dynamic_weights import DynamicWeightMatrix, MGAEmbedding
from history_feedback import MGANetwork

# 创建FastAPI应用
app = FastAPI(
    title="MGA-Net API",
    description="多粒度属性网络嵌入模型API服务",
    version="1.0.0"
)

# 定义请求模型
class WeightRequest(BaseModel):
    texts: List[str]  # 输入文本列表
    batch_size: Optional[int] = 4  # 批处理大小，默认4

# 定义响应模型
class WeightResponse(BaseModel):
    weights: List[List[float]]  # 权重矩阵
    text_prompts: List[str]  # 输入文本
    processing_time: float  # 处理时间(秒)

# 全局模型实例
model = None
clip_model = None
preprocess = None

@app.on_event("startup")
async def load_model():
    """启动时加载模型"""
    global model, clip_model, preprocess
    
    # 加载MGA-Net模型
    model = MGANetwork(device=DEVICE)
    model.eval()  # 设置为评估模式
    
    # 加载CLIP模型
    clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    print("模型加载完成，API服务已就绪")

@app.post("/mga-weight", response_model=WeightResponse)
# 优化建议1：添加类型注解
class WeightRequest(BaseModel):
    texts: List[str]
    batch_size: Optional[int] = Field(default=4, ge=1, le=8)  # 添加验证范围

# 优化建议2：使用更高效的批处理方式
async def get_mga_weights(request: WeightRequest):
    try:
        if not request.texts:
            raise HTTPException(status_code=400, detail="请提供至少一个文本描述")
        
        batch_size = min(request.batch_size, 8)
        start_time = time.time()
        
        # 使用列表推导式优化
        all_features = [
            clip_model.encode_text(
                clip.tokenize(request.texts[i:i+batch_size]).to(DEVICE)
            ).float()
            for i in range(0, len(request.texts), batch_size)
        ]
        
        # 处理文本提示，获取CLIP特征
        with torch.no_grad():
            # 分批处理文本以控制内存使用
            all_features = []
            for i in range(0, len(request.texts), batch_size):
                batch_texts = request.texts[i:i+batch_size]
                text_tokens = clip.tokenize(batch_texts).to(DEVICE)
                text_features = clip_model.encode_text(text_tokens).float()
                all_features.append(text_features)
                # 释放中间变量
                del text_tokens, text_features
                torch.cuda.empty_cache()
            
            # 合并所有批次的特征
            clip_features = torch.cat(all_features, dim=0)
            
            # 生成随机输入ID (模拟实际输入)
            vocab_size = 1000
            seq_length = 10
            input_ids = torch.randint(0, vocab_size, (len(request.texts), seq_length)).to(DEVICE)
            
            # 通过模型获取嵌入和权重
            embeddings = model.embedding(input_ids)
            weights = model.weight_matrix(embeddings.mean(dim=1), clip_features)
            
            # 将结果转换为numpy数组并返回
            weights_np = weights.cpu().numpy().tolist()
            
            # 计算处理时间
            processing_time = time.time() - start_time
            
            return {
                "weights": weights_np,
                "text_prompts": request.texts,
                "processing_time": processing_time
            }
            
    except Exception as e:
        # 记录错误并返回
        import traceback
        print(f"处理请求时出错: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)