# 3D-Wavelet-Decomposition-on-3D-MRI-Images-for-feature-extraction
Utilizing the notion of Wavelet Decomposition we decomposed the image into the mentioned frequency components by the action of the taken wavelet. After undergoing transformation, two types of wavelets emerge Father Wavelets, characterized by lower amplitude, and Mother Wavelets, characterised by higher amplitude. The values obtained from the mother wavelets are concatenated to produce the final coefficients, while the values obtained from the father wavelets contribute to the extraction of features. The pre-initialized weights were taken dot product with the retrieved features and coefficients, respectively. The weights were initialized according to the dimensions of the image, its channels, and the scaling factor. The width of the image was considered as the number of axes, with three wavelet modes being used. 
![image](https://github.com/pranava1709/3D-Wavelet-Decomposition-on-3D-MRI-Images-for-feature-extraction/assets/60814171/50a67dc2-6d39-4fa0-848a-04a1e55c3921)

The outcome of the dot product was disassembled into the revised frequency components and subsequently converted back to the time domain through the utilization of the Inverse Discrete Wavelet Transform (IDWT). Moreover, the initial image and the revised image were visually examined in order to assess the extent of information acquisition and perceptual lucidity, which might potentially serve as input for the Deep Learning framework. The complete code is written in PyTorch Framework.

![image](https://github.com/pranava1709/3D-Wavelet-Decomposition-on-3D-MRI-Images-for-feature-extraction/assets/60814171/5a64309f-eef6-4e8c-9549-c3c5ec5183b3)

