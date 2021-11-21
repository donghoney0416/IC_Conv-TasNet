# Inter-channel Conv-TasNet for multichannel speech enhancement

![Speech enhancement diagram](https://user-images.githubusercontent.com/93239188/142759283-881d2df3-778d-40d8-9cfc-5e6015cf5b02.png)

With the recent rising of live streaming services such as YouTube, outdoor recording is frequently observed in public. In particular, multichannel recording outdoor is commonly utilized to provide immersive audio to an audience beyond the screen. However, since outdoor recording inevitably entails ambient noise, ambient noise must be suppressed to make the desired speech heard well. In this work, we propose an end-to-end multichannel speech enhancement network that can handle inter-channel relationships on individual layers. In contrast to the conventional method, we build a network that can process data to maintain the various spatial information. Also, we made it possible for the network to process data from each perspective of feature and channel dimension. The proposed method outperforms the state-of-the-art multichannel variants of neural networks on the speech enhancement task, even with significantly smaller parameter sizes than the conventional methods.

![Clean speech](https://user-images.githubusercontent.com/93239188/142759349-ce638ff6-48c8-4bae-9eab-86462415a06b.png)
![Enhanced speech](https://user-images.githubusercontent.com/93239188/142759351-ab4035f5-4e36-4b7b-a5fe-8a24879ba157.png)
![Noisy speech](https://user-images.githubusercontent.com/93239188/142759352-f082b1b7-bc53-4d0a-9fa5-17bf420f72e6.png)
