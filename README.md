# neuralgae

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

> ./Neuralgia/image0.jpg: Gordon setter, minibus, sidewinder, miniskirt, hand-held computer, Doberman, steel drum, packet
> ./Neuralgia/image1.jpg: terrapin, comic book, worm fence, African chameleon, mousetrap, prayer rug, kimono, maillot
> ./Neuralgia/image2.jpg: common iguana, bullfrog, swimming trunks, bow, loggerhead, black grouse, horned viper, night snake
> ./Neuralgia/image3.jpg: fiddler crab, eft, tiger shark, green mamba, coho, chambered nautilus, electric ray, boa constrictor
> ./Neuralgia/image4.jpg: sleeping bag, coho, mongoose, coral reef, dhole, wood rabbit, frilled lizard, banded gecko
> ./Neuralgia/image5.jpg: frilled lizard, banded gecko, Airedale, diamondback, hyena, common iguana, partridge, terrapin


## Components

### deepdream.py

This is an adaptation of the Google Deepdreams ipython notebook.
I've added Audun's deepdraw technique to it

http://auduno.com/post/125362849838/visualizing-googlenet-classes

### drawing

dream.py
neuralgae_draw.sh      -> draw.sh
neuralgae_classify.py  -> classify.py
neuralgae.py           -> run_neuralgae.py

### writing

compile_defs.py
classes.txt
imagenet.py
generate.py
render.py
