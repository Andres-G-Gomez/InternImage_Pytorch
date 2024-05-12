# Custom InternImage Implementation

This repository contains a custom implementation of the InternImage model, focusing on modularity and ease of investigation into architectural alternatives. Unlike some existing implementations that rely on specific frameworks like mmsegmentation, this implementation is built solely on PyTorch, making it more flexible and accessible for experimentation.

## Contents

- **modules/**: This directory contains the core components of the InternImage model, including the encoder, the UperNet decoder, and supporting scripts necessary for running the model.
  
  - **modules/internimage.py**: Contains the implementation of the InternImage encoder, which is the backbone of the model. It includes the stem layer, InternImage blocks, and other necessary components.
  
  - **modules/upernet.py**: Contains the implementation of the UperNet decoder, responsible for processing the output of the encoder and generating the final segmentation masks.
  
- **tutorial/**: This directory contains tutorials to help users understand and utilize the core components of the InternImage model.

  - **tutorial/core_op_tutorial.ipynb**: A Jupyter notebook providing a comprehensive guide to the core operator of InternImage, DCNv3. It covers its principles, implementation details, and usage examples.
  
  - **tutorial/encoder_tutorial.ipynb**: A Jupyter notebook offering a detailed explanation of the InternImage encoder architecture. It covers the structure of the encoder, its components, and how to use it for various tasks.

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
