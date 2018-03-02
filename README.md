# STATUS

* **[Working]** lstm.py **Haven't been Tested yet !**

* **[Working]** main.py

# TODO List


### main --> Crear Main en Root

* **[Done]** Implementar Callback de Losswise en lstm.py

* Testear modelo en lstm.py

* Testear el main


### Testear archivos y clases

* **[Tested]** utils.py

* **[Tested]** perplexity.py

* **[Tested]** TokenSelector.py

* **[Not Tested]** Tokenization.py

* RNN.py

* **[Verified]** Generators.py  (**[Thread Safe]**)


### Crear archivo requirements.txt

* Hacer lista de paquetes y crear archivo requirements.txt


### perplexity

* **[Done]** Implementar perplexity [BPC] (bits per character)

* Implementar perplexity per word


### Clase Generators

* **[Done]** Verificar que clase Generators se ajusta al uso de L y Lprima


---


## Switch KERAS Backend to Theano

Keras Documentation: [Switching from one backend to another](https://keras.io/backend/#switching-from-one-backend-to-another)


```
cd ~/.keras/
nano keras.json

```

Edit keras.json, and change the field "backend" to "theano"

```
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "theano"
}
```

## Theano Configuration (GPU)

From [Theano Configuration](http://deeplearning.net/software/theano/library/config.html)

Create a config file at home dir

```
cd ~
nano .theanorc
```

And copy and paste this configuration

```
[global]
device = cuda
force_device = True
floatX = float32
mode=FAST_RUN
optimizer_including=cudnn

[dnn]
enabled = True
include_path = /usr/local/cuda/include
library_path = /usr/local/cuda/lib64

[gpuarray]
preallocate = 1

[lib]
cnmem = 1

[blas]
ldflags = -lmkl_rt -lpthread

```

And save it.

then, on a terminal, execute

```
export MKL_THREADING_LAYER=GNU
```
and press enter

or

mañana probamos -lpthread ( ver https://github.com/Theano/Theano/issues/5348 )

---

## clone_github.sh


Edit clone_github.sh

```
cd ~
nano clone_github.sh
```

And copy and paste this script

```
#!/bin/bash
cd ~
cd Syllables
chmod 777 -R syllable-aware/
rm -R syllable-aware/
git clone https://github.com/nlpchile/syllable-aware.git
```

And save it.


---
# Conda Environment


### List of conda environments

```
conda env list
```

### Activate Environment (base)

```
source activate base
```

### Theano Install

```
pip install theano --upgrade
```

### Keras Install

```
pip install keras --upgrade
```

### Deactivate Enviroment (base)

```
source deactivate base
```


## Extras

```
conda install mkl
conda install mkl-service
conda install openblas
conda install pygpu
```

---
# Screen

```
init screen
```

### Detaching From Screen

```
press ctrl + a then press d
```

### Attaching from last screen 

```
screen -r
```
if exists may screen this command show a list with the Ids

### Attaching specific screen

```
screen -D -r id_screen
```

---

# Data

## Syllable-aware tests

Spanish data for tests obtained from [HERE](https://github.com/yoonkim/lstm-char-cnn/blob/master/get_data.sh)


## Spanish Billion Words Corpus

Raw Data down obtained from [Spanish Billion Words Corpus](http://cs.famaf.unc.edu.ar/~ccardellino/SBWCE/clean_corpus.tar.bz2)

> Cristian Cardellino: Spanish Billion Words Corpus and Embeddings (March 2016), http://crscardellino.me/SBWCE/


## getData

```
wget http://cs.famaf.unc.edu.ar/~ccardellino/SBWCE/clean_corpus.tar.bz2

tar xf clean_corpus.tar.bz2

mv spanish_billion_words ./data/
```
