# Train your language model course

We have all used LLMs (Large Language Models) and we are impressed by their capabilities. I got curious about how these systems were built and I decided to learn this by creating this course. I am from Morocco and I speak the Moroccan darija. The current LLMs in the market understand this language a little bit but you can't have coherent conversations with them in Moroccan darija. As a challenge I wanted to train a language model from scratch on my WhatsApp conversations because they are in Moroccan darija.

I have created a YouTube playlist where I document what I did in this course, feel free to watch at your own pace and if something is not clear, open an issue in this repository and I will make sure to help you if I can.

[![course_thumbnail](./images/course_thumbnail%20.png)](https://www.youtube.com/playlist?list=PLMSb3cZXtIfptKdr56uEdiM5pR6HDMoUX)

## Repository contents

- `notebooks/`: Jupyter notebooks for each step in the pipeline.
- `slides/`: Presentation slides used in the YouTube series.
- `data/`: Dummy data and templates.
- `transformer/`: Contains the Transformer and LoRA scripts.
- `minbpe/`: Is the tokenizer that I took from [Andrej's](https://github.com/karpathy/minbpe) repository since it is not available as a package.

## Dependencies

You will need to install [Python](https://www.python.org/downloads/) and some dependencies, run the following command to install the dependencies:

```bash
pip install -r requirements.txt
```

## Course content

You will learn many concepts in this course:

1. Extracting data from WhatsApp.
2. Tokenizing the data with the BPE algorithm.
3. The Transformer architecture.
4. Pre-training stage.
5. Fine-tuning dataset.
6. Fine-tuning (Instruction fine-tuning and LoRA fine-tuning)

For each concept, there is a dedicated video in the [playlist](https://www.youtube.com/playlist?list=PLMSb3cZXtIfptKdr56uEdiM5pR6HDMoUX) and a dedicated notebook in the [notebooks](./notebooks/) folder.

## Contributions

We welcome contributions! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## Contact

For any questions or feedback, you can:

- Leave a comment on the YouTube videos.
- Connect with me on [LinkedIn](https://www.linkedin.com/in/imadsaddik/).
- Send me an email at [simad3647@gmail.com](mailto:simad3647@gmail.com).
