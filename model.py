from sentence_transformers import SentenceTransformer
import os

# 模型名称
model_name = 'sentence-transformers/all-mpnet-base-v2'
# 保存路径（对应仓库要求的目录）
save_path = 'E:\PycharmProjects\LinearRAG\model\\all-mpnet-base-v2'

print(f"正在从 Hugging Face 下载模型 {model_name}...")
model = SentenceTransformer(model_name)

print(f"正在保存模型到 {save_path}...")
if not os.path.exists('model'):
    os.makedirs('model')
model.save(save_path)

print("下载并保存完成！")