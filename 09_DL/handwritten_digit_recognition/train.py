import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(21)

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transforms = transforms.Compose([
    transforms.ToTensor(), # 将图像转换为张量
    transforms.Normalize([0.5], [0.5]) # 标准化图像数据
])

# 加载MNIST数据集 6W训练集 1W测试集
train_dataset = datasets.MNIST(root='./09_DL/handwritten_digit_recognition/data', train=True, download=True, transform=transforms) # 下载训练集
test_dataset = datasets.MNIST(root='./09_DL/handwritten_digit_recognition/data', train=False, download=True, transform=transforms) # 下载测试集

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # 对训练集进行打包，指定批次为64
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)  # 对测试集进行打包

class QYNet(nn.Module):
    def __init__(self):
        super().__init__()
         # 定义全连接层
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1) #28x28展平成一维 784
        x = torch.relu(self.fc1(x))  # 第一层 + ReLU激活
        x = torch.relu(self.fc2(x))  # 第二层 + ReLU激活
        x = torch.relu(self.fc3(x))  # 第三层 + ReLU激活
        x = self.fc4(x)  # 第四层（输出层，不需要激活函数）

# 初始化模型，并将模型移动到GPU上
model = QYNet().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器，学习率为0.001

# 训练模型
epochs = 10

for epoch in range(epochs):  # 训练10个epoch
    running_loss = 0.0
    correct_train = 0  # 正确预测的数量
    total_train = 0  # 样本总数

    # 训练过程
    model.train()  # 设定模型为训练模式