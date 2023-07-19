

# Audio Question Answering (AQA)

PyTorch code accompanies our Interspeech 2023 paper:

**Multi-Scale Attention for Audio Question Answering** \[[arXiv](https://arxiv.org/abs/2305.17993)\]

[Guangyao Li](https://ayameyao.github.io/), Yixin Xu and [Di Hu](https://dtaoo.github.io/index.html)

---

## Requirements

```python
python3.6 +
pytorch1.6.0
tensorboardX
ffmpeg
```

## Usage

1. **Clone this repo**

   ```python
   https://github.com/GeWu-Lab/MWAFM.git
   ```

2. **Download data**

   [Clotho-AQA](https://zenodo.org/record/6473207) and [AQA-MUSIC-AVQA](https://gewu-lab.github.io/MUSIC-AVQA/)
   
3. **Data pre-processing**

   We follow exact the same setting data format as [MUSIC AVQA](https://gewu-lab.github.io/MUSIC-AVQA/).

   **Notice:** We examined the original annotation files of Clotho-AQA and found that the official open-source annotations were not cleansed, resulting in discrepancies where different annotators provided different answers for the same question. As a result, we performed a simple filtering process where we considered a question to have the correct answer if it had at least two identical answers Based on this filtering process, we obtained a new and more accurate annotation file. The files in 'metadata' folder are described as follows

   - 'single_word\_[train/val/test].csv', Does not contain samples with answers *yes* and *no*.
   - 'single_word\_[train/val/test]\_clean.csv', Does not contain samples with answers *yes* and *no*. (Cleaned data)
   - 'clotho_aqa\_[train/val/test]\_clean.csv', Contains samples with answers *yes* and *no*. (Cleaned data)
   - 'binary\_[train/val/test]\_clean.csv', Include only samples with answers *yes* and *no*. (Cleaned data)

   

4. **Train and evaluate**

   Training

   ```python
   python main_MWAFM.py --mode train
   ```

   Testing

   ```python
   python main_MWAFM.py --mode test
   ```


## Citation

If you find this work useful, please consider citing it.

<pre><code>
@ARTICLE{Li2023MultiScale,
  title	= {Multi-Scale Attention for Audio Question Answering},
  author	= {Guangyao li, Yixin Xu, Di Hu},
  journal	= {Proc. INTERSPEECH},
  year	= {2023},
}
</code></pre>



## Acknowledgement

This research was supported by Public Computing Cloud, Renmin University of China.

