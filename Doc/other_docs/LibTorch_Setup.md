# Setup

## Libtorch without cmake

### 下载

[Libtorch官网](https://pytorch.org/get-started/locally/)

![image-20211123172134349](https://i.loli.net/2021/11/23/1vmu6Sc2bgNPoMJ.png)

- 注意：一个是release版本，一个debug版本

### 配置环境

![image-20211123172417660](https://i.loli.net/2021/11/23/AxjIG5zs7ad8cwl.png)

include文件夹是配置libtorch所需的头文件；

lib文件夹中有.lib与.dll两种文件；

**.lib路径要在vs中进行设置，.dll一般要添加到环境变量中**

- 环境变量配置

  右键我的电脑->属性->高级系统设置->高级中的环境变量->点击系统变量中的Path->添加dll路径：

  ![image-20211123172636695](https://i.loli.net/2021/11/23/LJ3Qf2WV6OEKD5C.png)

- Visual Studio 配置

  本人使用的是VS2019版本

  - **注意：**低版本的VS对C++新标准支持的程度可能比较低，在使用新版本库的时候可能会出现很多语法错误。

1. 配置管理器

   ![image-20211123173214068](https://i.loli.net/2021/11/23/R2Fb5zYKO3uvZ7f.png)

   使配置对应下载的版本，平台仅支持`x64`。

   ![image-20211123173249162](https://i.loli.net/2021/11/23/Y98o2eUASxFNCw3.png)

2. 设置头文件路径

   右键项目->属性

   ![image-20211123180404695](https://i.loli.net/2021/11/23/qgNurUGjophkmiX.png)

3. 设置链接库

   ![image-20211123184149840](https://i.loli.net/2021/11/23/CX3OnUqty2xNS6o.png)

   ```cpp
   c10.lib
   libprotobufd.lib
   mkldnn.lib
   torch.lib
   torch_cpu.lib
   ```

4. 添加库目录

   ![image-20211123184315992](https://i.loli.net/2021/11/23/ILzuAogx7QSdHwM.png)

### 测试

测试代码包含了最常用的两个头文件

```cpp
#include "torch/torch.h"
#include "torch/script.h"

int main()
{
    torch::Tensor output = torch::zeros({ 3,2 });
    std::cout << output << std::endl;

    return 0;
}
```

![image-20211123184742605](https://i.loli.net/2021/11/23/sK3kTgLljZE5Uxr.png)

### 问题

请不要自定义名为`T`、`test`等的宏，会冲突报错。