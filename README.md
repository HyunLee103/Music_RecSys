# 카카오 멜론 플레이리스트 추천시스템
- 2020년 여름에 진행된 카카오 아레나 대회에 참가하며 진행한 playlist continuation 프로젝트입니다.
- 플레이리스트에 수록된 곡과 태그의 절반 또는 전부가 숨겨져 있을 때, 주어지지 않은 곡들과 태그를 예측합니다.
- Public 상위 4% 랭크했습니다.
- https://arena.kakao.com/c/8

## Split data
1. First, put original `train.json` data in `res/` folder.  

2. Run `split_data.py`
	```bash
	>python split_data.py run res/train.json
	```

3. Check your directory
	```bash
	$> tree -d
	.
	├── arena_data (new directory!)
	│   ├── answers
	│   	├── val.json	# the answer to 'questions/val.json'
	│   ├── orig
	│   	├── train.json	# 80% original data
	│   	├── val.json	# 20% original data
	│   └── questions
	│   	├── val.json	# masked data of 'orig/val.json'
	│   
	└── res
	```

Train with `orig/train.json` and test with `questions/val.json`.  
After prediction, I recommend you to save the result as `results.json` and put that in `arena_data/results/` directory.  
Then you should run the evaluation code with `answers/val.json` and `results/results.json`.  
This is Arena official github style. I will follow this steps :)  

## Requirements
- pandas  
- numpy   
- sklearn  
- PyTorch  
- gensim  

## Usage
	python main.py

## Process
#### 1. CF, Sparse matrix -> 점수 rating(ply 담긴 순서) – 11만 x 63만 -> 500개(충분하게)

평가지표 함수 nDCG의 역함수로 가중치 함수 모델링, 최대값인 200(100 for val,test)까지 가중치가 1 미만으로 떨어지지 않게 조정. x : ply에 담긴 순서 

https://www.desmos.com/calculator/lrbcbfdqjr


  
#### 2. val, test : song, tags 빠진 것을 채워넣기

- title(x), song(O), tags(O) 4190 -> MF
- title(x), song(O), tags(x) 4507 -> ply 임배딩(song)/autoencoder -> 인접 ply tags rating -> MF 
- title(O) song(x), tags(O) 1232 -> title2song -> MF
- title(O) song(x), tags(x)  809(+2개는 다없음) -> title2tags, title2song -> MF
 
*100곡(CF) + 400곡(random) -> 셔플 -> rerank -> 100곡 유지정도  

 
#### 3. Rerank 
메타데이터 : Date, 장르, 가수, 플레이리스트 제목, tags, song 활용해서 부스팅으로 re-ranking
*reranking시 각 곡에 대한 Factorization machine으로 score 계산해보기
- Data 분포에 따른 가중곱   
- Title(word2vec) -> 장르(word2vec)이랑 비교해서 그 장르에 해당하는 노래에 가중치  
- Train set으로 artist density(unique score) 임계값 정하고(상위 n%) 이를 넘으면 가중치 
- Ply title autoencoder  (title2rec.py)

 


## Contributor
Hyelin Nam(<https://github.com/HyelinNAM>), Kyojung Koo(<https://github.com/koo616>) , Sanghyung Jung(<https://github.com/SangHyung-Jung>)  


