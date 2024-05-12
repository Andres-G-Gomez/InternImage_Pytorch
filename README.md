# Custom InternImage Implementation


This repository hosts a custom implementation of the InternImage model, designed with a focus on modularity and flexibility for investigating architectural alternatives. Leveraging PyTorch as its core framework, this implementation offers accessibility and ease of experimentation. It complements the insights provided in the article ["Navigating the InternImage Model - A Deep Dive into its Encoder and Revolutionary DCNv3 Operator"](https://andres-g-gomez.github.io/projects/5_project/) by Andres G. Gomez, offering practical exploration opportunities beyond theoretical discussions. For reference, the original InternImage repository can be found [here](https://github.com/OpenGVLab/InternImage).

## Contents

- **modules/**: This directory contains the core components of the InternImage model, including the encoder, the UperNet decoder, and supporting scripts necessary for running the model.
  
  - **internimage.py**: Contains the implementation of the InternImage encoder, which is the backbone of the model. It includes the stem layer, InternImage blocks, and other necessary components.
  
  - **upernet.py**: Contains the implementation of the UperNet decoder, responsible for processing the output of the encoder and generating the final segmentation masks.
  
- **tutorial/**: This directory contains tutorials to help users understand and utilize the core components of the InternImage model.

  - **core_op_tutorial.ipynb**: A Jupyter notebook providing a comprehensive guide to the core operator of InternImage, DCNv3. It covers its principles, implementation details, and usage examples.
  
  - **encoder_tutorial.ipynb**: A Jupyter notebook offering a detailed explanation of the InternImage encoder architecture. It covers the structure of the encoder, its components, and how to use it for various tasks.

## Usage

To use this custom InternImage implementation, follow these steps:

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/Andres-G-Gomez/InternImage_Pytorch.git
   ```
   
2. Install the required dependencies. Ensure you have PyTorch installed, as this implementation relies solely on PyTorch.
   ```bash
   pip install -r requirements.txt
   ```

3. Explore the modules and tutorials to understand the architecture and functionality of the InternImage model.

4. Utilize the provided scripts and examples to integrate the InternImage model into your projects or experiment with architectural variations.

## Contributing

Contributions to this project are welcome! If you find any issues, have suggestions for improvements, or want to contribute enhancements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
