### 1. The purpose of the project : 잇몸 검출

### 2. 대상 : 동일한 parameters 를 갖는 모델을 선정

### 3. 평가 : 단순한 Task & 개선된 모델 적용으로 인해 결과가 전반적으로 다 좋게 나왔다.

### 4. 모델 선택 : UnetPlusPlus & efficientnet-b5

### 5. 선택 이유 : 모델의 성능을 평가할 때 판단할 때 고려해야 하는 요소는 크게 3가지이다.

      1) 시간 (Latency & Throughput) : When the model infer the result, the shorter the time , the better the model. But, there is no substantial difference between Architectures & Encoders in this case.
                    So I think that the Latency & Throughput is not critical to determine the best model.
      2) 사용량 (CPU & GPU) : Just as Latency & Throughput, so there is no substantial difference in Usage of CPU and GPU. 
      3) 정확도 (Accuracy) : There is also no substantial difference in Accuracy.
                    But in this case (when there is no difference in Time, Usage and Accuracy), I think that it is better to focus on the Accuracy
                    In the Accuracy, we must concentrate on the distribution of data
                    The lower the variance, the better the model, in terms of performance of model.
                    So, the minimum value of accuracy is important.
                        We can improve the performance analyzing the pictures that have the minimum accuracy.
                        The fact that the variance is low means that the minimum vale is close to average.
                        Although the averages of accuracy are same, we trust the model that has the minimum vaule close to average.
      4) All things considered, I pick up the UnetPlusPlus & efficientnet-b5 model. 
