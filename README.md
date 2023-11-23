
### 项目简介

**项目名称：** 聋哑人手语语音翻译系统


**背景和国内外研究现状：**
聋哑人士依赖手语进行交流，这对非手语使用者构成了沟通上的挑战。目前，国内外一些研究机构和团队开始探索将手语翻译为语音的技术，但实时翻译和准确性仍然是挑战。

**项目目标：** 我们的目标是利用技术建立便利聋哑人的社区，能够准确将手语动作转化为口头语言，为聋哑人士提供一种更便捷、更自主的交流方式。通过这个系统，我们希望打破交流壁垒，促进包容和无障碍沟通的社会。

**关键特性：**
- 实时翻译：系统能够即时将手语动作转换为口头语言，实现翻译。
- 可拓展性：设计灵活的架构，以便未来能够快速扩展和集成新的功能和语言。
- 用户友好性：简单易用的界面，使得用户能够轻松上手使用这一技术。


**研究的内容和技术路线：**
我们的研究内容涉及关键点检测和语音合成。技术路线包括手语视频数据收集、数据标注、深度学习模型训练以及翻译系统的开发。

**创新创意点：**
在手语翻译系统中，我们并非通过图像分类去检测手语，我们使用的是更灵活的方式
，基于关键点检测通过一定的算法获取特征向量，将特征向量与向量数据库匹配获取词义，最后通过VITS实现语音合成
通过这种架构，无需再训练多个模型便可动态添加识别手语，让聋哑人士也能通过自己的方式创建属于自己的流行手语。
同时为未来集成新功能和语言提供可能性。

**应用领域：** 除了个人日常生活外，这项技术还有望在教育、医疗和商业领域发挥巨大潜力。它能够为聋哑人士提供更多教育资源和医疗服务，并在工作场所促进更有效的沟通。

**项目意义：** 这个项目不仅仅是技术上的突破，更是对社会包容和平等的重要贡献。通过改善聋哑人士的交流方式，我们致力于构建一个更加平等和包容的社会，让每个人都能够自由表达、平等交流。

**存在的问题以及今后的改进方向：**
准确性问题：优化提取特征向量的算法和优化检测模型以提高.翻译的精准度。