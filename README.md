# Deep Audio Dyslexia Detection

Dataset can be found [here](data/raw/scores).

Columns description:
* ID: participant ID
* grade: school grade (1-4)
* age: participant age
* gender: participant gender (0 - female, 1 - male)
* raven: Raven's progressive matrices score
* speed: reading speed (number of words read in 1 minute)
* comprehension (if present): number of correct answers to comprehension questions
* label1: dyslexia label based on reading speed only (0 - dyslexia, 1 - risk of dyslexia, 2 - typically developing reader)
* label2 (if present): dyslexia label based on reading speed and comprehension (0 - dyslexia, 1 - risk of dyslexia, 2 - typically developing reader)




External dependencies:
ffmpeg
python 3.12.11