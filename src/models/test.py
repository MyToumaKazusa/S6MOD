import torch
import torch.nn as nn
import torch.optim as optim

# 假设 SS2D 是您的模型类
model = SS2D()
model.train()  # 确保模型处于训练模式

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 假设有一个数据加载器 data_loader
for epoch in range(num_epochs):
    for inputs, labels in data_loader:
        # 将数据移动到 GPU（如果可用）
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')