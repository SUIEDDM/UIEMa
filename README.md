# UIEMa
A Hybrid Mamba and Sparse LUT Network for Perceptual-friendly Underwater Image Enhancement
## Abstract
Red light attenuation, medium absorption differences, and suspended particles are the main factors causing color distortion and detail blurring in underwater images. However, most existing deep learning methods face challenges in the enhancement process, such as high computational costs, insufficient global modeling, imprecise or excessive local adjustments, and a lack of perceptua-friendly color distribution. This article proposes a hybrid network for underwater image enhancement that combines Mamba and 3D sparse Look-up Table (LUT) to address the aforementioned issues. Mamba achieves interaction between global context and local information through four-directional and bidirectional cross-scanning, global coarse and local fine-grained features are effectively aggregated within the corresponding stages of the Encoder-Decoder, enabling image global modeling and restoration of local details at a lower computational cost. 3D sparse LUT compensates for color egradation through the mapping of color change matrices, making the overall quality more consistent with visual perception. Noteworthy improvements are showcased across three full and non-reference underwater benchmarks, our method achieves gains of up to 28.6dB and 0.98 on PSNR and SSIM compared to twelve state-of-the-art models, effectively correcting color distortion while improving texture details. 

![image](https://github.com/SUIEDDM/UIEMa/blob/main/fig2.png)

## Download
 
You can download the datasets on [UIEB](https://li-chongyi.github.io/proj_benchmark.html)(UIEB),[LSUI](https://lintaopeng.github.io/_pages/UIE%20Project%20Page.html) and [U45](https://github.com/IPNUISTlegal/underwater-test-dataset-U45-)

