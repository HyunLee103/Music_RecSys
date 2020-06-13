# Kakao_music_recommend
Organize study and code preparing for the Kakao (melon) Playlist Song and Tag Recommendation Contest.  

## Team member
Hyelin Nam(<https://github.com/HyelinNAM>), Kyojung Koo(<https://github.com/koo616>) , Sanghyung Jung(<https://github.com/SangHyung-Jung>)  


## Process
### 1.     CF, Sparse matrix -> 점수 rating(ply 담긴 순서) – 11만 x 63만 -> 500개(nangnanglly)

평가지표 함수 nDCG의 역함수로 가중치 함수 모델링, 최대값인 200(100 for val,test)까지 가중치가 1 미만으로 떨어지지 않게 조정. x : ply에 담긴 순서  



https://www.desmos.com/calculator/lrbcbfdqjr


  
  
### 2.     val, test : song, tags 빠진 것을 채워넣고  

2.1. title(x), song(O), tags(O) 4190 -> CF (혜린)  

2.2. title(x), song(O), tags(x) 4507 -> ply 임배딩(song)/autoencoder -> 인접 ply tags rating -> CF (상형)  

2.3. title(O) song(x), tags(O) 1232 -> title2song -> CF (교정)  

2.4. title(O) song(x), tags(x)  809(+2개는 다없음) -> title2tags, title2song -> CF (현)  

 
*100곡(CF) + 400곡(random) -> 셔플 -> rerank -> 100곡 유지정도  

 
### 3.     Rerank : 메타데이터 : Date, 장르, 가수, 플레이리스트 제목, tags, song
*reranking시 각 곡에 대한 Factorization machine으로 score 계산해보기
3.1. Date 분포에 따른 간드러지는 가중곱  

3.2. Title(word2vec) -> 장르(word2vec)이랑 비교해서 그 장르에 해당하는 노래에 가중치  

3.3. Train set으로 artist density(unique score) 임계값 정하고(상위 n%) 이를 넘으면 가중치 (장르에도 적용해봐요)  

3.4. Ply title autoencoder  

 
*학습 시 val set은 train에서 split(베이스라인 코드 참고)

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


