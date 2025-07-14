#控制几个脚本依次执行

#!/bin/bash

# 执行第一个脚本
./tailor.sh

# 检查第一个脚本的返回代码，如果不为 0，停止执行
if [ $? -ne 0 ]; then
    echo "Script 1 failed."
    exit 1
fi

# 等待 30 秒
echo "Waiting for 30 seconds..."
sleep 30

# 执行第二个脚本
./train.sh

# 同样，检查第二个脚本的返回代码
if [ $? -ne 0 ]; then
    echo "Script 2 failed."
    exit 1
fi

# 等待 30 秒
echo "Waiting for another 30 seconds..."
sleep 30

# 执行第三个脚本
./output.sh

# 检查第三个脚本的返回代码
if [ $? -ne 0 ]; then
    echo "Script 3 failed."
    exit 1
fi

echo "All scripts executed successfully."
