## Semi-supervised image classification using perturbed images 

### overview 
1) why semi-supervised learninig ?
    21세기 데이터의 가치는 점점 높아지고 있고, 데이터는 점점 더 많아지고 있다. 한편, 레이블링된 데이터는 그 양이 제한적이며 휴먼 레이버한 작업이다. 특히 전문적인 도메인의 데이터는 레이블링 작업에 특히나 전문 인력이 필요하게 되고, 그는 또 막대한 비용을 필요로 한다. 만약 모델이 레이블링 되지 않은 데이터를 사용하여 학습할 수 있다면, 혹은 적은 레이블의 데이터만을 사용하여 학습할 수 있다면, 앞에서 언급한 비용을 줄이면서 데이터의 활용성을 높일 수 있을 것이다. 그리고 사실상, 사람은 레이블이 없어도 학습을 한다. 우리의 인공지능 모델 또한 레이블이 없는 환경에서도 그 개념을 학습할 수 있어야 한다고 생각한다. 

2) pseudo labeling and consistency regularization term 
    현재 semi-supervised image classification task의 몇몇 데이터셋에서 SOTA를 달성하고 있는 FixMatch[1] 모델의 경우, pseudo labeling과 consistency regularization term을 통한 learning framework을 통해 좋은 성능을 달성하는 것을 볼 수 있다. 모델이 labeled example을 통해 representation을 학습해 나가고, hyperparameter로 결정된 threshold를 통해 unlabeled example을 학습에 사용하는 비중을 매 iteration마다 결정한다. unlabeled example을 활용할 때, waekly augmented(Crop, Flip, Shift) example과 strongly augmented(RandAugment + Cutout) example 사이의 KL divergence term을 loss로 활용한다. 즉, input image가 조금 파악하기 어려워지더라도, 같은 이미지이면 같은 분포를 가지도록 학습을 하게 만드는 것이다. 

### proposed model 
    나는 pseudo labeling 과 consistency regaularization term이 모두 결국은 좋은 representation 을 학습하는 기술적인 방법으로서 이용된다고 생각했다. 좋은 represetation을 학습시키는 것이 관건이고, FixMatch framework에서 더 다양하게 input image의 변형을 주어 consistency regularization term 을 학습에 활용할 수 있다면 더 좋은 represetation 을 학습하게 할 수 있지 않을까? 라고 생각하게 되었다. 따라서 본 프로젝트에서는 strong augmented image를 대체할 수 있는, pertuerbed image generation network 을 학습하여 모델에게 다양한 augmented image를 주어 좋은 representation을 가지게 하는 것을 목표로 하였다. 

### results and analysis 
### future plans 
### reference 
