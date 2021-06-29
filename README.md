# nlpForBeginner_zyh

## Task-1：文本分类
### Dataset： sentiment-analysis-on-movie-reviews
### Result：
 | Model | Accuracy |	Others |
 | --- | --- | --- |
 | Bert-base |	53.93%	|	53.2% |
 | Glove+RNN |	62.88%	|	43.6% |

## Task-2： 文本匹配
### Dataset： snli_1.0
### Result:
 | Model |	Accuracy	| Others |
 | --- | --- | --- |
 | Bi-LSTM	| 79.19% |	84.5% |
 | Bert-base	| 81.24% |  |		

## Task-3: 序列标注
### Dataset: CoNLL-2003
### Result: 
 | Model	| Accuracy	| Others |
 | ---| --- | --- |
 | Bert-base	| 92.39%	|	92.4% |

## Task-4: 古诗生成
### Dataset: PoetryFromTang.txt / poetry1000.json
### Result:
 | Model |	Train_PPL |	Valid_PPL	| Example |
 | ---| --- | --- | --- |
 | GPT-2 | 1.07 | 11.91 | 梅年高年一相春，一年相人有山山。<br> 兰年春雨雨，春家花色烟。<br>竹日高高一情情，一林林人有山林。<br>菊無無事事， 閑閑思思閑。
 | GPT2-Chinese | 1.06 | 9.97 | 梅花滿翠雲，山前有一峰。<br> 兰將軍家，今朝已不如。<br> 竹花香依依依，一枝獨自長。<br> 菊滿江山，一峰春自海邊 |
 | Seq2Seq |  |  |  |
 | GRU |  |  |  |
      
      
