# RNN-LSTM
RNN &amp; LSTM Text Generation Programming Project for CSE 151B (UCSD)

For this project, I trained and fine-tuned RNN and LSTM models on Shakespearean-style text. The data/tiny_shakespeare.txt file contains 40,000 lines from a mix of his plays. The models were trained to predict the next character given a sequence of characters, like the RNN/LSTM model shown below:

<p align="center">
 <img width="171" height="203" alt="Pasted Graphic" src="https://github.com/user-attachments/assets/d9505581-510a-4256-b0a7-85ee6a2bc144" />
</p>

A final cross-entropy loss of 1.29 was achived using the LSTM model with 15 epochs, a sequence length of 32, and 2 hidden layers each of size 150 neurons.

## Generated Text Samples:

### Output 1:

***As which I remember you to shame the reason boy of nor no souls, if they are at liberty,
She shall not say true let him come from her lord, and by me to lie and stooping in me:
Nor one hunger by the people will come to see hope must be spent to see, the streams present garden men,
And thou wert too determine. Who in my joys are high and feeling to bid the warrant for that?***

***CLARENCE:
I like a comfort is a soldiers; no talk off to fell him at your queen's house,
Like in charge is to watch your birth; he came to bed, their accusation from the sight
That it with the carveries crowns become myself in anguishment. These most royalties be their heavy aider be,
No doubts, well I lay and trust such a while. So-night, believe your proper lamentation!
Thou hast a boldness of law confusion to pardon thee to the watch of their tenarting streams
Of thy company in hand: and so sins you to his hands. Come, come, let's come: we will be so, from her senators doth it but begun more than this.***

### Output 2:

_**Macbeth
 by William Shakespeare
 Edited by Barbara A. Mowat and Paul Werstiness**_

***ROMEO:
Ay, bring so? Farewell, come, Signior king;
That a master,
And he had safety,' the grace's daughter is as that's unto you,
And bed, too trymeth with his mere.***

***GONZALO:
The time so in journey, man, good Barnardie,' good, lord,
For this answer in what
Our as smothers, come, I am pass.***

***BUCKINGHAM:
And it is a people, and headst so own dog,
And slept not friend: wherefore!--***

***ISABELLA:
The princess, that's more.***

***LUCIO:
Madam, so, thou as, then, nay, that he lives
Upon Eviged, and thou would remember her
Claudio made the treatures of Angelo.***

***JULIET:
Yet, look it, stamp you.***

***Servant:
What, wilt he would be you, not me: for vain too,
And officillury now to struck theind spoils?
What thou begen of yourself; what is off a mystery.
And see the assuiff! you shall be
But and a leave hateness to it. We
One too will, by the mother?***


## Files:

```rnn.py``` : RNN model architecture. <br>
```lstm.py``` : LSTM model architecture. <br>
```shakespeare_dataset.py``` : Custom dataset class used to create dataloaders. <br>
```train.py``` : Training loop and evaluation function. <br>
```util.py``` : Contains helper functions for data processing and plotting loss curves. <br>
```main.py``` : Initialize model with config parameters and call train and eval functions. <br>
```generate.py``` : Function to generate text with pre-trained LSTM or RNN model. <br>
```config.py``` : Function to load .yaml configuration files. <br>
```data/tiny_shakespeare.txt``` : Shakespeare play data. <br>


