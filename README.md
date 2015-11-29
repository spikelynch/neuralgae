# Neuralgae

My original idea for NaNoGenMo15 was to try to apply an analogy of the
deepdreams technique to written texts. Deepdreams works by
taking a neural net which has been trained to discriminate between
image categores, and applying it to an existing image with the 'gain'
turned up, so that it amplifies the patterns it sees and hallucinates
shapes arising from the categories it has been trained on. I thought
it should be possible to use a neural net which had been trained to
discriminate between different kinds of text, and apply the same
forced-feedback technique so that it would hallucinate texts from a
randomised starting point.

This idea turned out to be beyond my abilities as a neural-net
programmer - I've gotten pretty good at abusing them to create
deepdreams, but have a lot to learn about desigining and training
them.

Along the way, I started tinkering with
[Audun Øygard's deepdraw technique](http://auduno.com/post/125362849838/visualizing-googlenet-classes),
which uses a different technique to produce dreamlike visualisations
of single image categories which look quite unlike the acid-trippy
deepdreams hallucinations. I'd adapted his code to allow
visualisations of hybrid categories - maximising two or more
categories rather than just one - and it occured to me that a
two-stage feedback process could generate a sequence of images which
were linked, as follows:

* given a set S of categories, use deepdraw to render an image
* use the neural net to classify the image and generate a new set S'
of categories

The second set S' could then be fed back to deepdraw to generate the
next image. Hopefully, the images would have some visual resemblance
to one another, giving a kind of pseudo-narrative flow, but avoiding a
situation where the visualisation got 'stuck' on a limited set of
categories.

This tendency to get stuck affected all of the early test runs - the
algorithm would start off with an interesting random-looking
visualisation, but after a few iterations would gravitate towards
those image categories which the particular parameters of the base
image and the deepdraw renderer reinforced. Lobsters, nautiluses and
snakelike or wormy shapes were really common attractors.

I got around this problem by adding a bit of randomness to the
category selection process. Instead of taking the top N image
categories from the classifier, and then rendering an image from
those, the control script takes N categories, where N is fairly large
(100 is the value I settled on), then picks a random sample of S of those
categories, where S is a lot smaller (8 worked well). This still gives
some continuity between frames, but the random sampling helps the
algorithm move out of local maxima in what's a really huge space of
potential images.

There are still some image classes which act as attractors - worms,
jellyfish, snails and other marine invertebrates are really common
classes, and every so often the algorithm will pass through the very
large number of dog breed categories - but the algorithm doesn't get
trapped in them anymore. The underwater feel of a lot of the images
made me change the title from "Neuralgia" to "Neuralgae".

This was all good fun, but NaNoGenMo is about generating texts, not
images: I thought a thousand weird dreamlike images could form the
basis of a graphic novel or illustrated narrative. The framework for
the text was the list of image classes used to generate each frame:

    ./Neuralgia/image0.jpg: Gordon setter, minibus, sidewinder, miniskirt, hand-held computer, Doberman, steel drum, packet
    ./Neuralgia/image1.jpg: terrapin, comic book, worm fence, African chameleon, mousetrap, prayer rug, kimono, maillot
    ./Neuralgia/image2.jpg: common iguana, bullfrog, swimming trunks, bow, loggerhead, black grouse, horned viper, night snake
    ./Neuralgia/image3.jpg: fiddler crab, eft, tiger shark, green mamba, coho, chambered nautilus, electric ray, boa constrictor
    ./Neuralgia/image4.jpg: sleeping bag, coho, mongoose, coral reef, dhole, wood rabbit, frilled lizard, banded gecko
    ./Neuralgia/image5.jpg: frilled lizard, banded gecko, Airedale, diamondback, hyena, common iguana, partridge, terrapin

At first I wrote a script to pull the first few words of the top entry
for a Google search on adjacent terms, but this felt a bit
scattershot, and many of the searches returned copies of the ImageNet
category list in the first place. I had been playing around with the
Python NLTK toolkit as a way to generate texts, which led me to
realise that the list of 1000 image categories which the deepdreams
neural net had been trained on - part of an annual image recognition
tournament called ImageNet - was taken from
[WordNet](http://wordnet.princeton.edu/), a popular on-line lexical
database.

WordNet is a collection of "synsets" (synonym sets) which roughly
speaking map onto a concept. Each of the ImageNet categories is a
synset in WordNet, and WordNet also includes a definition of each
synset. I decided to use NLTK to generate a dictionary for the 1000
ImageNet categories:

    tench freshwater dace-like game fish of Europe and western Asia noted for ability to survive outside water
    goldfish small golden or orange-red freshwater fishes of Eurasia used as pond or aquarium fishes
    great_white_shark large aggressive shark widespread in warm seas; known to attack humans
    tiger_shark large dangerous warm-water shark with striped or spotted body
    hammerhead a stupid person; these words are used to express a low opinion of someone's intelligence
    hammerhead the striking part of a hammer
    hammerhead medium-sized live-bearing shark with eyes at either end of a flattened hammer-shaped head; worldwide in warm waters; can be dangerous

Note that I included all of the different synsets attached to a single
word, such as 'hammerhead'.  This dictionary gave be a source text
which was guaranteed to have every word in the list of image classes,
and I could then put together a simple Markov chain algorithm to
generate a line of 'verse' for the eight classes belonging to each
image:

![image](https://raw.githubusercontent.com/spikelynch/neuralgae/master/images/image0.jpg)

    Gordon setter a diamond pattern into limestone
    minibus a government agency or grain
    sidewinder small short-legged terrier originally by
    miniskirt a hanging clusters of the
    hand-held computer a fighting ax; used to
    Doberman medium for military style raincoat;
    steel drum a device that displays them
    packet a long thick coarse weedy

Markov text generation - or "ebooks", as it's become known on
Twitter, after the famous @horse_ebooks account, which turned out to
not be a bot after all - is a bit done to death, but I think that the
qualities of the source text here make for fairly decent automated
surrealism. Combined with the images, it's surprisingly readable.

I'm still in the process of formatting an HTML version of the results:
I'll be posting the URL when it's ready.

## Dependencies

The code requires the
[Caffe deep learning framework](http://caffe.berkeleyvision.org/) with
the Python interface pycaffe.

It also needs the
[CaffeNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet)
and
[GoogLeNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet)
neural models - I had to use both because for some reason I couldn't
work out, GoogLeNet always segfaults when I use it to classify images
on my Mac. I ended up using CaffeNet to do the classification and
GoogLeNet to generate the images (I like GoogLeNet's output better).

It also requires the following software:

* the ImageMagick image processing and generation package
* Python (2.7) and the following Python packages
  - NLTK
  - pystache

There are several more Python dependencies required to get Caffe
running - these are described in detail on the Caffe website.
  
## Credits

* Princeton University (2010), [WordNet](http://wordnet.princeton.edu)
  Princeton University
* Bird, Steven, Edward Loper and Ewan Klein (2009), [Natural Language Processing with Python.](http://www.nltk.org/) O’Reilly Media Inc.
* Øygard, Audun (2015), ['Visualising GoogLeNet classes'](http://auduno.com/post/125362849838/visualizing-googlenet-classes)
* Mordvintsev, Alexander, Tyka, Michael and Olah, Christopher (2015)
['deepdream GitHub repository'](https://github.com/google/deepdream)
