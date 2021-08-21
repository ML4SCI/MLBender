<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->


<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/SinanGncgl/Background-Estimation-NormalizingFlows">
    <img src="images/gsoc.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">MLBender</h3>
  <h3 align="center">Background Estimation with Neural Autoregressive Flows</h3>

  <p align="center">
    A research project that uses Normalizing Flows to perform Background Estimation Task. (This project was carried out for GSoC-2021 (Google Summer of Code) with ML4SCI (Machine Learning for Science), <a href="https://summerofcode.withgoogle.com/projects/4726444372000768">Click Here for more details.</a>).
    <br />
    <a href="Documentation.md"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    ·
    <a href="https://github.com/SinanGncgl/Background-Estimation-NormalizingFlows/issues">Report Bug</a>
    ·
    <a href="https://github.com/SinanGncgl/Background-Estimation-NormalizingFlows/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Dataset](#dataset)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)
* [Special Thanks](#special-thanks)



<!-- ABOUT THE PROJECT -->
## About The Project

<img src="images/project.jpeg" alt="Logo">

Data-driven background estimation is crucial for many scientific searches, including searches for new phenomena in experimental datasets. Neural autoregressive flows (NAF) is a deep generative model that can be used for general transformations and is therefore attractive for this application. The MLBENDER project focuses on studying how to develop such transformations that can be learned and applied to a region of interest. In this project the main aim is implementing a Neural Autoregressive Flow (NAF) model to estimate the background distribution and apply it to a representative physics analysis searching for a resonance excess over a smooth background.

### Built With
* [Pytorch](https://pytorch.org/)
* [Python](https://www.python.org/)



<!-- GETTING STARTED -->
## Getting Started

* All project related informations can be found in [main.ipynb](main.ipynb) file.

### Prerequisites

It is recommended to use conda or venv environments if you're going to run this on your local PC.

### Installation

Main packages are as follows:
* PyTorch 1.9
* Python 3.6 or higher
* Pandas
* Seaborn
* numpy
* sklearn


<!-- USAGE EXAMPLES -->
## Dataset

* Dataset can be found at: [LHC Dataset](https://zenodo.org/record/2629073).
* You can work with the other datasets if you like, the normalizing flow models are suitable to work with any other datasets.


<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/SinanGncgl/Background-Estimation-NormalizingFlows/issues) for a list of proposed features (and known issues).



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE.txt -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Sinan Gençoğlu - [@SinanGncgl](https://www.linkedin.com/in/sinan-gencoglu/) - sinan.gencogluu@gmail.com

Project Link: [https://github.com/SinanGncgl/Background-Estimation-NormalizingFlows](https://github.com/SinanGncgl/Background-Estimation-NormalizingFlows)


<!-- SPECIAL THANKS -->
## Special Thanks
* [Sergei Glyzer](http://sergeigleyzer.com/)
* [Meenakshi Narain](https://twitter.com/meenakshinarain)
* [Emanule Usai](https://emanueleusai.com/)
* [Aneesh Heintz]()
* [Muhammad Ehsan ul Haq](https://twitter.com/Ehsanbinejaz)
