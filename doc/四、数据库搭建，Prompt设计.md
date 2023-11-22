# 一、知识库文档处理

 ## 1.1、知识库设计

选用一些经典开源课程、视频（部分）作为示例， 包括PDF， MP4， md格式的资料。

知识库源数据放置在 ../../data_base/knowledge_db 目录下。

## 1.2、文档加载

### 1.2.1、PDF文档

使用PyMuPDFLoader 来读取知识库的 PDF 文件。PyMuPDFLoader 是 PDF 解析器中速度最快的一种，结果会包含 PDF 及其页面的详细元数据，并且每页返回一个文档。

```
## 安装必要的库
# !pip install rapidocr_onnxruntime -i https://pypi.tuna.tsinghua.edu.cn/simple
# !pip install "unstructured[all-docs]" -i https://pypi.tuna.tsinghua.edu.cn/simple
# !pip install pyMuPDF -i https://pypi.tuna.tsinghua.edu.cn/simple

from langchain.document_loaders import PyMuPDFLoader

# 创建一个 PyMuPDFLoader Class 实例，输入为待加载的 pdf 文档路径
loader = PyMuPDFLoader("../../data_base/knowledge_db/pumkin_book/pumpkin_book.pdf")

# 调用 PyMuPDFLoader Class 的函数 load 对 pdf 文件进行加载
pages = loader.load()

```

**数据探索**

文档加载后储存在 pages 变量中:

page 的变量类型为 List
打印 pages 的长度可以看到 pdf 一共包含多少页

```
print(f"载入后的变量类型为：{type(pages)}，",  f"该 PDF 一共包含 {len(pages)} 页")
-> 载入后的变量类型为：<class 'list'>， 该 PDF 一共包含 196 页

page = pages[1]
print(f"每一个元素的类型：{type(page)}.", 
    f"该文档的描述性数据：{page.metadata}", 
    f"查看该文档的内容:\n{page.page_content[0:1000]}", 
    sep="\n------\n")

每一个元素的类型：<class 'langchain.schema.document.Document'>.
------
该文档的描述性数据：{'source': '../../data_base/knowledge_db/pumkin_book/pumpkin_book.pdf', 'file_path': '../../data_base/knowledge_db/pumkin_book/pumpkin_book.pdf', 'page': 1, 'total_pages': 196, 'format': 'PDF 1.5', 'title': '', 'author': '', 'subject': '', 'keywords': '', 'creator': 'LaTeX with hyperref', 'producer': 'xdvipdfmx (20200315)', 'creationDate': "D:20230303170709-00'00'", 'modDate': '', 'trapped': ''}
------
查看该文档的内容:
前言
“周志华老师的《机器学习》（西瓜书）是机器学习领域的经典入门教材之一，周老师为了使尽可能多的读
者通过西瓜书对机器学习有所了解, 所以在书中对部分公式的推导细节没有详述，但是这对那些想深究公式推
导细节的读者来说可能“不太友好”，本书旨在对西瓜书里比较难理解的公式加以解析，以及对部分公式补充
具体的推导细节。”
读到这里，大家可能会疑问为啥前面这段话加了引号，因为这只是我们最初的遐想，后来我们了解到，周
老师之所以省去这些推导细节的真实原因是，他本尊认为“理工科数学基础扎实点的大二下学生应该对西瓜书
中的推导细节无困难吧，要点在书里都有了，略去的细节应能脑补或做练习”。所以...... 本南瓜书只能算是我
等数学渣渣在自学的时候记下来的笔记，希望能够帮助大家都成为一名合格的“理工科数学基础扎实点的大二
下学生”。
```
page 中的每一元素为一个文档，变量类型为 langchain.schema.document.Document, 文档变量类型包含两个属性

- page_content 包含该文档的内容。
- meta_data 为文档相关的描述性数据。

### 1.2.2、MD文档

读入 markdown 文档：
```
from langchain.document_loaders import UnstructuredMarkdownLoader

loader = UnstructuredMarkdownLoader("../../data_base/knowledge_db/prompt_engineering/1. 简介 Introduction.md")
pages = loader.load()

```

读取对象：
```
print(f"载入后的变量类型为：{type(pages)}，",  f"该 Markdown 一共包含 {len(pages)} 页")
载入后的变量类型为：<class 'list'>， 该 Markdown 一共包含 1 页
page = pages[0]
print(f"每一个元素的类型：{type(page)}.", 
    f"该文档的描述性数据：{page.metadata}", 
    f"查看该文档的内容:\n{page.page_content[0:]}", 
    sep="\n------\n")
每一个元素的类型：<class 'langchain.schema.document.Document'>.
------
该文档的描述性数据：{'source': '../../data_base/knowledge_db/prompt_engineering/1. 简介 Introduction.md'}
------
查看该文档的内容:
第一章 简介

欢迎来到面向开发者的提示工程部分，本部分内容基于吴恩达老师的《Prompt Engineering for Developer》课程进行编写。《Prompt Engineering for Developer》课程是由吴恩达老师与 OpenAI 技术团队成员 Isa Fulford 老师合作授课，Isa 老师曾开发过受欢迎的 ChatGPT 检索插件，并且在教授 LLM （Large Language Model， 大语言模型）技术在产品中的应用方面做出了很大贡献。她还参与编写了教授人们使用 Prompt 的 OpenAI cookbook。我们希望通过本模块的学习，与大家分享使用提示词开发 LLM 应用的最佳实践和技巧。

网络上有许多关于提示词（Prompt， 本教程中将保留该术语）设计的材料，例如《30 prompts everyone has to know》之类的文章，这些文章主要集中在 ChatGPT 的 Web 界面上，许多人在使用它执行特定的、通常是一次性的任务。但我们认为，对于开发人员，大语言模型（LLM） 的更强大功能是能通过 API 接口调用，从而快速构建软件应用程序。实际上，我们了解到 DeepLearning.AI 的姊妹公司 AI Fund 的团队一直在与许多初创公司合作，将这些技术应用于诸多应用程序上。很兴奋能看到 LLM API 能够让开发人员非常快速地构建应用程序。

在本模块，我们将与读者分享提升大语言模型应用效果的各种技巧和最佳实践。书中内容涵盖广泛，包括软件开发提示词设计、文本总结、推理、转换、扩展以及构建聊天机器人等语言模型典型应用场景。我们衷心希望该课程能激发读者的想象力，开发出更出色的语言模型应用。

随着 LLM 的发展，其大致可以分为两种类型，后续称为基础 LLM 和指令微调（Instruction Tuned）LLM。基础LLM是基于文本训练数据，训练出预测下一个单词能力的模型。其通常通过在互联网和其他来源的大量数据上训练，来确定紧接着出现的最可能的词。例如，如果你以“从前，有一只独角兽”作为 Prompt ，基础 LLM 可能会继续预测“她与独角兽朋友共同生活在一片神奇森林中”。但是，如果你以“法国的首都是什么”为 Prompt ，则基础 LLM 可能会根据互联网上的文章，将回答预测为“法国最大的城市是什么？法国的人口是多少？”，因为互联网上的文章很可能是有关法国国家的问答题目列表。

与基础语言模型不同，指令微调 LLM 通过专门的训练，可以更好地理解并遵循指令。举个例子，当询问“法国的首都是什么？”时，这类模型很可能直接回答“法国的首都是巴黎”。指令微调 LLM 的训练通常基于预训练语言模型，先在大规模文本数据上进行预训练，掌握语言的基本规律。在此基础上进行进一步的训练与微调（finetune），输入是指令，输出是对这些指令的正确回复。有时还会采用RLHF（reinforcement learning from human feedback，人类反馈强化学习）技术，根据人类对模型输出的反馈进一步增强模型遵循指令的能力。通过这种受控的训练过程。指令微调 LLM 可以生成对指令高度敏感、更安全可靠的输出，较少无关和损害性内容。因此。许多实际应用已经转向使用这类大语言模型。


```

### 1.2.3、MP4视频
对本地 MP4 视频进行处理，需要首先经过转录加载成文本格式，在加载到 LangChain 中。使用 Whisper 实现视频的转写，详见教程：  [知乎|开源免费离线语音识别神器whisper如何安装](https://zhuanlan.zhihu.com/p/595691785)

使用 Whisper 在原目录下输出转写结果：
```
whisper ../../data_base/knowledge_db/easy_rl/强化学习入门指南.mp4 --model large --model_dir whisper-large --language zh --output_dir ../../data_base/knowledge_db/easy_rl

```
注意，此处 model_dir 参数应是下载到本地的 large-whisper 参数路径。

转化完后，会在原目录下生成强化学习入门指南.txt 文件，直接加载该 txt 文件即可：

```
from langchain.document_loaders import UnstructuredFileLoader
loader = UnstructuredFileLoader("../../data_base/knowledge_db/easy_rl/强化学习入门指南.txt")
pages = loader.load()

```

内容读取：
```
page = pages[0]
print(f"每一个元素的类型：{type(page)}.", 
    f"该文档的描述性数据：{page.metadata}", 
    f"查看该文档的内容:\n{page.page_content[0:1000]}", 
    sep="\n------\n")

```

## 1.3、文档分割

Langchain 中文本分割器都根据 chunk_size (块大小)和 chunk_overlap (块与块之间的重叠大小)进行分割。

![image.png](https://upload-images.jianshu.io/upload_images/7289495-f6deba96c570cc28.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- chunk_size 指每个块包含的字符或 Token （如单词、句子等）的数量

- chunk_overlap 指两个块之间共享的字符数量，用于保持上下文的连贯性，避免分割丢失上下文信息

Langchain 提供多种文档分割方式，区别在怎么确定块与块之间的边界、块由哪些字符/token组成、以及如何测量块大小

- RecursiveCharacterTextSplitter(): 按字符串分割文本，递归地尝试按不同的分隔符进行分割文本。
- CharacterTextSplitter(): 按字符来分割文本。
- MarkdownHeaderTextSplitter(): 基于指定的标题来分割markdown 文件。
- TokenTextSplitter(): 按token来分割文本。
- SentenceTransformersTokenTextSplitter(): 按token来分割文本
- Language(): 用于 CPP、Python、Ruby、Markdown 等。
- NLTKTextSplitter(): 使用 NLTK（自然语言工具包）按句子分割文本。
- SpacyTextSplitter(): 使用 Spacy按句子的切割文本。

```
''' 
* RecursiveCharacterTextSplitter 递归字符文本分割
RecursiveCharacterTextSplitter 将按不同的字符递归地分割(按照这个优先级["\n\n", "\n", " ", ""])，
    这样就能尽量把所有和语义相关的内容尽可能长时间地保留在同一位置
RecursiveCharacterTextSplitter需要关注的是4个参数：

* separators - 分隔符字符串数组
* chunk_size - 每个文档的字符数量限制
* chunk_overlap - 两份文档重叠区域的长度
* length_function - 长度计算函数
'''
#导入文本分割器
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 知识库中单段文本长度
CHUNK_SIZE = 500

# 知识库中相邻文本重合长度
OVERLAP_SIZE = 50

# 此处使用 PDF 文件作为示例
from langchain.document_loaders import PyMuPDFLoader

# 创建一个 PyMuPDFLoader Class 实例，输入为待加载的 pdf 文档路径
loader = PyMuPDFLoader("../../data_base/knowledge_db/pumkin_book/pumpkin_book.pdf")

# 调用 PyMuPDFLoader Class 的函数 load 对 pdf 文件进行加载
pages = loader.load()
page = pages[1]

# 使用递归字符文本分割器
from langchain.text_splitter import TokenTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=OVERLAP_SIZE
)
text_splitter.split_text(page.page_content[0:1000])

-> ['前言\n“周志华老师的《机器学习》（西瓜书）是机器学习领域的经典入门教材之一，周老师为了使尽可能多的读\n者通过西瓜书对机器学习有所了解, 所以在书中对部分公式的推导细节没有详述，但是这对那些想深究公式推\n导细节的读者来说可能“不太友好”，本书旨在对西瓜书里比较难理解的公式加以解析，以及对部分公式补充\n具体的推导细节。”\n读到这里，大家可能会疑问为啥前面这段话加了引号，因为这只是我们最初的遐想，后来我们了解到，周\n老师之所以省去这些推导细节的真实原因是，他本尊认为“理工科数学基础扎实点的大二下学生应该对西瓜书\n中的推导细节无困难吧，要点在书里都有了，略去的细节应能脑补或做练习”。所以...... 本南瓜书只能算是我\n等数学渣渣在自学的时候记下来的笔记，希望能够帮助大家都成为一名合格的“理工科数学基础扎实点的大二\n下学生”。\n使用说明\n• 南瓜书的所有内容都是以西瓜书的内容为前置知识进行表述的，所以南瓜书的最佳使用方法是以西瓜书\n为主线，遇到自己推导不出来或者看不懂的公式时再来查阅南瓜书；\n• 对于初学机器学习的小白，西瓜书第 1 章和第 2 章的公式强烈不建议深究，简单过一下即可，等你学得',
 '有点飘的时候再回来啃都来得及；\n• 每个公式的解析和推导我们都力 (zhi) 争 (neng) 以本科数学基础的视角进行讲解，所以超纲的数学知识\n我们通常都会以附录和参考文献的形式给出，感兴趣的同学可以继续沿着我们给的资料进行深入学习；\n• 若南瓜书里没有你想要查阅的公式，或者你发现南瓜书哪个地方有错误，请毫不犹豫地去我们 GitHub 的\nIssues（地址：https://github.com/datawhalechina/pumpkin-book/issues）进行反馈，在对应版块\n提交你希望补充的公式编号或者勘误信息，我们通常会在 24 小时以内给您回复，超过 24 小时未回复的\n话可以微信联系我们（微信号：at-Sm1les）；\n配套视频教程：https://www.bilibili.com/video/BV1Mh411e7VU\n在线阅读地址：https://datawhalechina.github.io/pumpkin-book（仅供第 1 版）\n最新版 PDF 获取地址：https://github.com/datawhalechina/pumpkin-book/re']


split_docs = text_splitter.split_documents(pages)
print(f"切分后的文件数量：{len(split_docs)}")

-> 切分后的文件数量：737
print(f"切分后的字符数（可以用来大致评估 token 数）：{sum([len(doc.page_content) for doc in split_docs])}")

-> 切分后的字符数（可以用来大致评估 token 数）：314712

```


## 1.4、文档词向量化

在机器学习和自然语言处理（NLP）中，Embeddings（嵌入）是一种将类别数据，如单词、句子或者整个文档，转化为实数向量的技术。这些实数向量可以被计算机更好地理解和处理。嵌入背后的主要想法是，相似或相关的对象在嵌入空间中的距离应该很近。

举个例子，我们可以使用词嵌入（word embeddings）来表示文本数据。在词嵌入中，每个单词被转换为一个向量，这个向量捕获了这个单词的语义信息。例如，"king" 和 "queen" 这两个单词在嵌入空间中的位置将会非常接近，因为它们的含义相似。而 "apple" 和 "orange" 也会很接近，因为它们都是水果。而 "king" 和 "apple" 这两个单词在嵌入空间中的距离就会比较远，因为它们的含义不同。

使用方法：
- 直接使用 openai 的模型生成 embedding：openAI 的模型需要消耗 api，对于大量的token 来说成本会比较高，但是非常方便。

- 使用 HuggingFace 上的模型生成 embedding： HuggingFace 的模型可以本地部署，可自定义合适的模型，可玩性较高，但对本地的资源有部分要求。

- 采用其他平台的 api。对于获取 openAI key 不方便的可以采用这种方法。

使用示例：

```
# 使用前配置自己的 api 到环境变量中如
import os
import openai
import zhipuai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv()) # read local .env fileopenai.api_key  = os.environ['OPENAI_API_KEY']
openai.api_key  = os.environ['OPENAI_API_KEY']
zhihuai.api_key = os.environ['ZHIPUAI_API_KEY']

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from zhipuai_embedding import ZhipuAIEmbeddings

# embedding = OpenAIEmbeddings() 
# embedding = HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")
embedding = ZhipuAIEmbeddings()

```

```
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

query1 = "机器学习"
query2 = "强化学习"
query3 = "大语言模型"

# 通过对应的 embedding 类生成 query 的 embedding。
emb1 = embedding.embed_query(query1)
emb2 = embedding.embed_query(query2)
emb3 = embedding.embed_query(query3)

# 将返回结果转成 numpy 的格式，便于后续计算
emb1 = np.array(emb1)
emb2 = np.array(emb2)
emb3 = np.array(emb3)

print(f"{query1} 生成的为长度 {len(emb1)} 的 embedding , 其前 30 个值为： {emb1[:30]}") 

机器学习 生成的为长度 1024 的 embedding , 其前 30 个值为： [-0.02768379  0.07836673  0.1429528  -0.1584693   0.08204    -0.15819356
 -0.01282174  0.18076552  0.20916627  0.21330206 -0.1205181  -0.06666514
 -0.16731478  0.31798768  0.0680017  -0.13807729 -0.03469152  0.15737721
  0.02108428 -0.29145902 -0.10099868  0.20487919 -0.03603597 -0.09646764
  0.12923686 -0.20558454  0.17238656  0.03429411  0.1497675  -0.25297147]

```

现在已经生成了对应的向量，如何度量文档和问题的相关性呢？

两种常用的方法：

- 计算两个向量之间的点积。 
- 计算两个向量之间的余弦相似度

点积是将两个向量对应位置的元素相乘后求和得到的标量值。点积相似度越大，表示两个向量越相似。

```
print(f"{query1} 和 {query2} 向量之间的点积为：{np.dot(emb1, emb2)}")
print(f"{query1} 和 {query3} 向量之间的点积为：{np.dot(emb1, emb3)}")
print(f"{query2} 和 {query3} 向量之间的点积为：{np.dot(emb2, emb3)}")

-> 机器学习 和 强化学习 向量之间的点积为：17.218882120572722
机器学习 和 大语言模型 向量之间的点积为：16.522186236712727
强化学习 和 大语言模型 向量之间的点积为：11.368461841901752

```

点积：计算简单，快速，不需要进行额外的归一化步骤，但丢失了方向信息。

余弦相似度：可以同时比较向量的方向和数量级大小。

余弦相似度将两个向量的点积除以它们的模长的乘积。其基本的计算公式为
![image.png](https://upload-images.jianshu.io/upload_images/7289495-b73d8f5fde9a2824.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

余弦函数的值域在-1到1之间，即两个向量余弦相似度的范围是[-1, 1]。当两个向量夹角为0°时，即两个向量重合时，相似度为1；当夹角为180°时，即两个向量方向相反时，相似度为-1。即越接近于 1 越相似，越接近 0 越不相似。

```
print(f"{query1} 和 {query2} 向量之间的余弦相似度为：{cosine_similarity(emb1.reshape(1, -1) , emb2.reshape(1, -1) )}")
print(f"{query1} 和 {query3} 向量之间的余弦相似度为：{cosine_similarity(emb1.reshape(1, -1) , emb3.reshape(1, -1) )}")
print(f"{query2} 和 {query3} 向量之间的余弦相似度为：{cosine_similarity(emb2.reshape(1, -1) , emb3.reshape(1, -1) )}")

机器学习 和 强化学习 向量之间的余弦相似度为：[[0.68814796]]
机器学习 和 大语言模型 向量之间的余弦相似度为：[[0.63382724]]
强化学习 和 大语言模型 向量之间的余弦相似度为：[[0.43555894]]

```
已经完成了文档的基本处理，向量数据库可以帮我们快速的管理和计算生成的 embedding 并寻找和 query 最相关的内容。

# 二、向量数据库的介绍及使用

## 2.1、向量数据库简介

向量数据库是用于高效计算和管理大量向量数据的解决方案。向量数据库是一种专门用于存储和检索向量数据（embedding）的数据库系统。它与传统的基于关系模型的数据库不同，它主要关注的是向量数据的特性和相似性。

在向量数据库中，数据被表示为向量形式，每个向量代表一个数据项。这些向量可以是数字、文本、图像或其他类型的数据。向量数据库使用高效的索引和查询算法来加速向量数据的存储和检索过程。

Langchain 集成了超过 30 个不同的向量存储库。选择 Chroma 是因为它轻量级且数据存储在内存中，非常容易启动和开始使用。

```
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from zhipuai_embedding import ZhipuAIEmbeddings

from langchain.llms import OpenAI
from langchain.llms import HuggingFacePipeline
from zhipuai_llm import ZhipuAILLM

# 使用前配置自己的 api 到环境变量中如
import os
import openai
import zhipuai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv()) # read local .env fileopenai.api_key  = os.environ['OPENAI_API_KEY']
openai.api_key  = os.environ['OPENAI_API_KEY']
zhipuai.api_key = os.environ['ZHIPUAI_API_KEY']

os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

# 加载 PDF
loaders_chinese = [
    PyMuPDFLoader("../../data_base/knowledge_db/pumkin_book/pumpkin_book.pdf") # 南瓜书
    # 可以自行加入其他文件
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

#md
folder_path = "../../data_base/knowledge_db/prompt_engineering/"
files = os.listdir(folder_path)
loaders = []
for one_file in files:
    loader = UnstructuredMarkdownLoader(os.path.join(folder_path, one_file))
    loaders.append(loader)
for loader in loaders:
    docs.extend(loader.load())

#mp4-txt
loaders = [
    UnstructuredFileLoader("../../data_base/knowledge_db/easy_rl/强化学习入门指南.txt") # 机器学习,
]
for loader in loaders:
    docs.extend(loader.load())


# 切分文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
split_docs = text_splitter.split_documents(docs)


# 定义 Embeddings
embedding = OpenAIEmbeddings() 
# embedding = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
# embedding = ZhipuAIEmbeddings()


persist_directory = '../../data_base/vector_db/chroma'

!rm -rf '../../data_base/vector_db/chroma'  # 删除旧的数据库文件（如果文件夹中有文件的话），window电脑请手动删除

```

## 2.2、构建 Chroma 向量库

```
vectordb = Chroma.from_documents(
    documents=split_docs[:100], # 为了速度，只选择了前 100 个切分的 doc 进行生成。
    embedding=embedding,
    persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
)

```

运行 vectordb.persist 来持久化向量数据库
```
vectordb.persist(）
```

或者直接加载已经构建好的向量库

```
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

print(f"向量库中存储的数量：{vectordb._collection.count()}")
向量库中存储的数量：1120
```

## 2.3、通过向量数据库检索

**相似度检索**

```
question="什么是机器学习"
sim_docs = vectordb.similarity_search(question,k=3)
print(f"检索到的内容数：{len(sim_docs)}")
->  检索到的内容数：3
for i, sim_doc in enumerate(sim_docs):
    print(f"检索到的第{i}个内容: \n{sim_doc.page_content[:200]}", end="\n--------------\n")
-> 检索到的第0个内容: 
导，同时也能体会到这三门数学课在机器学习上碰撞产生的“数学之美”。
1.1
引言
本节以概念理解为主，在此对“算法”和“模型”作补充说明。“算法”是指从数据中学得“模型”的具
体方法，例如后续章节中将会讲述的线性回归、对数几率回归、决策树等。“算法”产出的结果称为“模型”，
通常是具体的函数或者可抽象地看作为函数，例如一元线性回归算法产出的模型即为形如 f(x) = wx + b
的一元一次函数。
--------------
检索到的第1个内容: 
模型：机器学习的一般流程如下：首先收集若干样本（假设此时有 100 个），然后将其分为训练样本
（80 个）和测试样本（20 个），其中 80 个训练样本构成的集合称为“训练集”，20 个测试样本构成的集合
称为“测试集”，接着选用某个机器学习算法，让其在训练集上进行“学习”（或称为“训练”），然后产出
得到“模型”（或称为“学习器”），最后用测试集来测试模型的效果。执行以上流程时，表示我们已经默
--------------
检索到的第2个内容: 
→_→
欢迎去各大电商平台选购纸质版南瓜书《机器学习公式详解》
←_←
第 1 章
绪论
本章作为“西瓜书”的开篇，主要讲解什么是机器学习以及机器学习的相关数学符号，为后续内容作
铺垫，并未涉及复杂的算法理论，因此阅读本章时只需耐心梳理清楚所有概念和数学符号即可。此外，在
阅读本章前建议先阅读西瓜书目录前页的《主要符号表》，它能解答在阅读“西瓜书”过程中产生的大部
分对数学符号的疑惑。
本章也作为
--------------
```
**MMR 检索**

只考虑检索出内容的相关性会导致内容过于单一，可能丢失重要信息。

最大边际相关性 (MMR, Maximum marginal relevance) 可以帮助我们在保持相关性的同时，增加内容的丰富度。

核心思想是在已经选择了一个相关性高的文档之后，再选择一个与已选文档相关性较低但是信息丰富的文档。这样可以在保持相关性的同时，增加内容的多样性，避免过于单一的结果。

```
mmr_docs = vectordb.max_marginal_relevance_search(question,k=3)

for i, sim_doc in enumerate(mmr_docs):
    print(f"MMR 检索到的第{i}个内容: \n{sim_doc.page_content[:200]}", end="\n--------------\n")

MMR 检索到的第0个内容: 
导，同时也能体会到这三门数学课在机器学习上碰撞产生的“数学之美”。
1.1
引言
本节以概念理解为主，在此对“算法”和“模型”作补充说明。“算法”是指从数据中学得“模型”的具
体方法，例如后续章节中将会讲述的线性回归、对数几率回归、决策树等。“算法”产出的结果称为“模型”，
通常是具体的函数或者可抽象地看作为函数，例如一元线性回归算法产出的模型即为形如 f(x) = wx + b
的一元一次函数。
--------------
MMR 检索到的第1个内容: 
而人工智能的基本挑战是

学习在不确定的情况下做出好的决策

这边我举个例子

比如你想让一个小孩学会走路

他就需要通过不断尝试来发现

怎么走比较好

怎么走比较快

强化学习的交互过程可以通过这张图来表示

强化学习由智能体和环境两部分组成

在强化学习过程中

智能体与环境一直在交互

智能体在环境中获取某个状态后

它会利用刚刚的状态输出一个动作

这个动作也被称为决策

然后这个动作会
--------------
MMR 检索到的第2个内容: 
与基础语言模型不同，指令微调 LLM 通过专门的训练，可以更好地理解并遵循指令。举个例子，当询问“法国的首都是什么？”时，这类模型很可能直接回答“法国的首都是巴黎”。指令微调 LLM 的训练通常基于预训练语言模型，先在大规模文本数据上进行预训练，掌握语言的基本规律。在此基础上进行进一步的训练与微调（finetune），输入是指令，输出是对这些指令的正确回复。有时还会采用RLHF（reinforce
--------------

```

# 三、Prompt设计的原则和技巧
设计高效 Prompt 的两个关键原则：**编写清晰、具体的指令**和**给予模型充足思考时间**。

## 3.1、编写清晰、具体的指令
使用分隔符清晰地表示输入的不同部分

使用gpt-3.5-turbo示例：
```

import openai
import os
from dotenv import load_dotenv, find_dotenv


# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
#_ = load_dotenv(find_dotenv())

# 获取环境变量 OPENAI_API_KEY
openai.api_key = os.environ['OPENAI_API_KEY']
# 一个封装 OpenAI 接口的函数，参数为 Prompt，返回对应结果


def get_completion(prompt,
                   model="gpt-3.5-turbo"
                   ):
    '''
    prompt: 对应的提示词
    model: 调用的模型，默认为 gpt-3.5-turbo(ChatGPT)。你也可以选择其他模型。
           https://platform.openai.com/docs/models/overview
    '''

    messages = [{"role": "user", "content": prompt}]

    # 调用 OpenAI 的 ChatCompletion 接口
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )

    return response.choices[0].message["content"]

# 使用分隔符(指令内容，使用 ``` 来分隔指令和待总结的内容)
prompt = f"""
总结用```包围起来的文本，不超过30个字：


# 调用OpenAI
response = get_completion(prompt)
print(response)

```

## 3.2、寻求结构化的输出

什么是结构化输出呢？就是按照某种格式组织的内容，例如JSON、HTML等。这种输出非常适合在代码中进一步解析和处理。例如，您可以在 Python 中将其读入字典或列表中。

```
prompt = f"""
请生成包括书名、作者和类别的三本虚构的、非真实存在的中文书籍清单，\
并以 JSON 格式提供，其中包含以下键:book_id、title、author、genre。
"""
response = get_completion(prompt)
print(response)

{
  "books": [
    {
      "book_id": 1,
      "title": "迷失的时光",
      "author": "张三",
      "genre": "科幻"
    },
    {
      "book_id": 2,
      "title": "幻境之门",
      "author": "李四",
      "genre": "奇幻"
    },
    {
      "book_id": 3,
      "title": "虚拟现实",
      "author": "王五",
      "genre": "科幻"
    }
  ]
}

```

## 3.3、要求模型检查是否满足条件

如果任务包含不一定能满足的假设（条件），可以告诉模型先检查这些假设，如果不满足，则会指出并停止执行后续的完整流程。还可以考虑可能出现的边缘情况及模型的应对，以避免意外的结果或错误发生。

## 3.4、提供少量示例

例如：

```
prompt = f"""
您的任务是以一致的风格回答问题（注意：文言文和白话的区别）。
<学生>: 请教我何为耐心。
<圣贤>: 天生我材必有用，千金散尽还复来。
<学生>: 请教我何为坚持。
<圣贤>: 故不积跬步，无以至千里；不积小流，无以成江海。骑骥一跃，不能十步；驽马十驾，功在不舍。
<学生>: 请教我何为孝顺。
"""
response = get_completion(prompt)
print(response)

-> <圣贤>: 孝顺者，孝为本也。孝者，敬爱父母，尊重长辈，顺从家规，尽心尽力为家庭着想之道也。孝顺者，行孝之人，不忘亲恩，不辜负父母之养育之恩，以孝心感恩报答，尽己之力，尽孝之道。

```

## 3.5、指定完成任务所需的步骤

我们描述了杰克和吉尔的故事，并给出提示词执行以下操作：

首先，用一句话概括三个反引号限定的文本。
第二，将摘要翻译成英语。
第三，在英语摘要中列出每个名称。
第四，输出包含以下键的 JSON 对象：英语摘要和人名个数。要求输出以换行符分隔。

```
text = f"""
在一个迷人的村庄里，兄妹杰克和吉尔出发去一个山顶井里打水。\
他们一边唱着欢乐的歌，一边往上爬，\
然而不幸降临——杰克绊了一块石头，从山上滚了下来，吉尔紧随其后。\
虽然略有些摔伤，但他们还是回到了温馨的家中。\
尽管出了这样的意外，他们的冒险精神依然没有减弱，继续充满愉悦地探索。
"""

prompt = f"""
1-用一句话概括下面用<>括起来的文本。
2-将摘要翻译成英语。
3-在英语摘要中列出每个名称。
4-输出一个 JSON 对象，其中包含以下键：English_summary，num_names。
请使用以下格式：
文本：<要总结的文本>
摘要：<摘要>
翻译：<摘要的翻译>
名称：<英语摘要中的名称列表>
输出 JSON：<带有 English_summary 和 num_names 的 JSON>
Text: <{text}>
"""

response = get_completion(prompt)
print("prompt :")
print(response)


->prompt :
1-用一句话概括下面用<>括起来的文本：兄妹在迷人的村庄里冒险，遇到了意外但依然充满愉悦地探索。
2-将摘要翻译成英语：In a charming village, siblings Jack and Jill set off to fetch water from a well on top of a hill. While singing joyfully, they climb up but unfortunately, Jack trips on a stone and tumbles down the hill, with Jill following closely behind. Despite some minor injuries, they make it back to their cozy home. Despite the mishap, their adventurous spirit remains undiminished as they continue to explore with delight.
3-在英语摘要中列出每个名称：Jack, Jill
4-输出一个 JSON 对象，其中包含以下键：English_summary，num_names：
{
  "English_summary": "In a charming village, siblings Jack and Jill set off to fetch water from a well on top of a hill. While singing joyfully, they climb up but unfortunately, Jack trips on a stone and tumbles down the hill, with Jill following closely behind. Despite some minor injuries, they make it back to their cozy home. Despite the mishap, their adventurous spirit remains undiminished as they continue to explore with delight.",
  "num_names": 2
}

```


# 四、基于问答助手的Prompt构建

## 4.1、加载向量数据库

```
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings    # 调用 OpenAI 的 Embeddings 模型
import openai
from dotenv import load_dotenv, find_dotenv
import os

#import panel as pn # GUI
# pn.extension()


```
从环境变量中加载 OPENAI_API_KEY

```
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']
```

加载向量数据库

```
# 定义 Embeddings
embedding = OpenAIEmbeddings() 

# 向量数据库持久化路径
persist_directory = '../../data_base/vector_db/chroma'

# 加载数据库
vectordb = Chroma(
    persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
    embedding_function=embedding
)

```

测试一下加载的向量数据库，使用一个问题 query 进行向量检索。如下代码会在向量数据库中根据相似性进行检索，返回前 k 个最相似的文档。

用相似性搜索前，请确保你已安装了 OpenAI 开源的快速分词工具 tiktoken 包：pip install tiktoken

```
question = "什么是强化学习"
docs = vectordb.similarity_search(question,k=3)
print(f"检索到的内容数：{len(docs)}")

-> 检索到的内容数：3

for i, doc in enumerate(docs):
    print(f"检索到的第{i}个内容: \n {doc.page_content[:200]}", end="\n--------------\n")

```

## 4.2、创建一个 LLM

```
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature = 0 )
llm.predict("你好")

```

## 4.3、构建 prompt

```
from langchain.prompts import PromptTemplate

# template = """基于以下已知信息，简洁和专业的来回答用户的问题。
#             如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分。
#             答案请使用中文。
#             总是在回答的最后说“谢谢你的提问！”。
# 已知信息：{context}
# 问题: {question}"""
template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
{context}
问题: {question}
有用的回答:"""

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                 template=template)

# 运行 chain


from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})

```

创建检索 QA 链的方法 RetrievalQA.from_chain_type() 有如下参数：

- llm：指定使用的 LLM
- 指定 chain type : RetrievalQA.from_chain_type(chain_type="map_reduce")，也可以利用load_qa_chain()方法指定chain type。
- 自定义 prompt ：通过在RetrievalQA.from_chain_type()方法中，指定chain_type_kwargs参数，而该参数：chain_type_kwargs = {"prompt": PROMPT}
- 返回源文档：通过RetrievalQA.from_chain_type()方法中指定：return_source_documents=True参数；也可以使用RetrievalQAWithSourceChain()方法，返回源文档的引用（坐标或者叫主键、索引）


## 4.4、prompt 效果测试

```
question_1 = "什么是南瓜书？"
question_2 = "王阳明是谁？"

```
基于召回结果和 query 结合起来构建的 prompt 效果

```
result = qa_chain({"query": question_1})
print("大模型+知识库后回答 question_1 的结果：")
print(result["result"])

-> 大模型+知识库后回答 question_1 的结果：
南瓜书是对《机器学习》（西瓜书）中难以理解的公式进行解析和补充推导细节的一本书。谢谢你的提问！

result = qa_chain({"query": question_2})
print("大模型+知识库后回答 question_2 的结果：")
print(result["result"])

-> 大模型+知识库后回答 question_2 的结果：
我不知道王阳明是谁，谢谢你的提问！

```

大模型自己回答的效果

```
prompt_template = """请回答下列问题:
                            {}""".format(question_1)

### 基于大模型的问答
llm.predict(prompt_template)

-> "南瓜书是指《深入理解计算机系统》（Computer Systems: A Programmer's Perspective）一书的俗称。这本书是由Randal E. Bryant和David R. O'Hallaron合著的计算机科学教材，旨在帮助读者深入理解计算机系统的工作原理和底层机制。南瓜书因其封面上有一个南瓜图案而得名，被广泛用于大学的计算机科学和工程课程中。"


prompt_template = """请回答下列问题:
                            {}""".format(question_2)

### 基于大模型的问答
llm.predict(prompt_template)

->  '王阳明（1472年-1529年），字仲明，号阳明子，是明代中期著名的思想家、政治家、军事家和教育家。他提出了“心即理”、“知行合一”的思想，强调人的内心自觉和道德修养的重要性。他的思想对中国历史产生了深远的影响，被后世尊称为“阳明先生”。'

```


# 五、添加历史对话的记忆功能

## 5.1. 记忆（Memory）

使用 LangChain 中的储存模块，ConversationBufferMemory ，它保存聊天消息历史记录的列表，这些历史记录将在回答问题时与问题一起传递给聊天机器人，从而将它们添加到上下文中。

```
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
    return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
)

```

## 5.2、对话检索链（ConversationalRetrievalChain）

对话检索链（ConversationalRetrievalChain）在检索 QA 链的基础上，增加了处理对话历史的能力。

它的工作流程是:

- 将之前的对话与新问题合并生成一个完整的查询语句。
- 在向量数据库中搜索该查询的相关文档。
- 获取结果后,存储所有答案到对话记忆区。
- 用户可在 UI 中查看完整的对话流程。

```
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings    # 调用 OpenAI 的 Embeddings 模型
import openai
from dotenv import load_dotenv, find_dotenv
import os

_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

# 定义 Embeddings
embedding = OpenAIEmbeddings() 
# 向量数据库持久化路径
persist_directory = '../../data_base/vector_db/chroma'
# 加载数据库
vectordb = Chroma(
    persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
    embedding_function=embedding
)

# 创建LLM
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature = 0 )


from langchain.chains import ConversationalRetrievalChain

retriever=vectordb.as_retriever()

qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory
)
question = "我可以学习到关于强化学习的知识吗？"
result = qa({"question": question})
print(result['answer'])

-> 是的，根据提供的上下文，这门课程会教授关于强化学习的知识。


question = "为什么这门课需要教这方面的知识？"
result = qa({"question": question})
print(result['answer'])

-> 这门课需要教授关于强化学习的知识，是因为强化学习是一种用来学习如何做出一系列好的决策的方法。在人工智能领域，强化学习的应用非常广泛，可以用于控制机器人、实现自动驾驶、优化推荐系统等。学习强化学习可以帮助我们理解和应用这一领域的核心算法和方法，从而更好地解决实际问题。

```

