# cyber-physical-systems
Shared tools for my research in cyber physical systems.

# Datasets

## Air Quality
### [Urban Air](https://www.microsoft.com/en-us/research/project/urban-air/)
[Direct download](https://www.microsoft.com/en-us/research/uploads/prod/2016/02/Data-1.zip)


### [Beijing](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/USXCAK)
[Direct download](https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/USXCAK/E4SLK5)

## [Physionet](https://physionet.org/content/challenge-2019/1.0.0/training/)
2019 Challenge raw data
```bash
mkdir -p data/p19 && cd data/p19
wget -r -N -c -np -nH --cut-dirs=4 https://physionet.org/files/challenge-2019/1.0.0/training/
```