# CSDN_DEBUG MCP SERVER

🟢 本项目主要针对现有大模型无法立刻发现的python配置、环境等问题。本人实践经验中，偶尔会出现大模型无法解决的报错、但是csdn一搜就能得到正确解决方案。因此，本项目通过对CSDN等各大平台数据的获取，作为知识库向量查询，构建了`CSDN_DEBUG` MCP服务。

🔴 本项目的测试问题是由大模型生成。因此测试的数据并不能满足这个项目需求，因此欢迎各位投稿大模型无法立刻解决的bug报错问题，将作为测试数据来进行前后分析对比。这是本人第一次开源项目，还有许多可以交流的地方，欢迎一起构建一个更加完善的`CSDN_DEBUG` MCP服务。

## 🎯 主要功能

🤖 ***获取csdn语料信息***

* 获取文章内容
* 文章去重
* 标题近似向量回归

***🔧 压缩向量***

* 内容去重（20%）
* 句子级抽取式摘要
* MMR 保多样性，控制每文token 上限与句数上限
* 跨文章融合

## 📦 安装

`step1`: 安装所需依赖

```cmd
pip install -r requirments.txt
```

`step2`: 配置

```json
{
  "mcpServers": {
    "csdn-helper": {
      "command": "your_python.exe",
      "args": ["-u ", "/root/path/server.py"]
    }
  }
}
```
`step3`: 你也可以通过mcp来启动你的服务
```python
mcp run yourserver.py
```
## 📞 联系方式

电子邮箱：yirongzzz@163.com



