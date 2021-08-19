<!--
*** Thanks for checking out this README Template. If you have a suggestion that would
*** make this better, please fork the repo and create a pull request or simply open
*** an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
-->





<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/Ehsan1997/NormalizingFlows_HEP">
    <img src="readme_stuff/coding.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">MLBender</h3>
  <h3 align="center">Normalizing Flows for Anomaly Detection in High Energy Physics</h3>

  <p align="center">
    A research project that uses Masked Autoregressive Flows to perform Anomaly detection on data from Large Hadron Collider. (This project was carried out for GSoC-2021 (Google Summer of Code) with ML4SCI (Machine Learning for Science), <a href="https://summerofcode.withgoogle.com/projects/#4624155581874176">Click Here for more details.</a>).
    <br />
    <a href="readme_stuff/Documentation.md"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://colab.research.google.com/drive/1UjTLf7a1995erDQH4x7b9VkTQrv0XS_S?authuser=1#scrollTo=eEZQ-qXuFOB0">View Demo</a>
    ·
    <a href="https://github.com/Ehsan1997/NormalizingFlows_HEP/issues">Report Bug</a>
    ·
    <a href="https://github.com/Ehsan1997/NormalizingFlows_HEP/issues">Request Feature</a>
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
* [Acknowledgements](#acknowledgements)
* [Special Thanks](#special-thanks)



<!-- ABOUT THE PROJECT -->
## About The Project

<!--[![Product Name Screen Shot][product-screenshot]](https://example.com)-->
![Product Name Screen Shot][product-screenshot]

Scientific Experiments yield abundant amount of data.
Traditionally processing this data requires a lot of supervision from different entities.
Advent of Deep Generative techniques in the field of anomaly detection have yielded quite satisfactory results.
Thus, in this project we use one of the latest proposed deep generative technique known as normalizing flows for anomaly detection in High Energy Physics data.

### Built With
* [Pytorch](https://pytorch.org/)
* [Python](https://www.python.org/)



<!-- GETTING STARTED -->
## Getting Started

This project is divided into two different parts.
* Colab Notebooks: In order to make use of GPU acceleration colab notebooks were used.
* Current Repository: It contains code for machine learning models and utilities.

Please check this [google drive folder](https://drive.google.com/drive/u/1/folders/1WKKG1v3bnAGqs82a4B1lOrI6Y1S0YAXO) for notebooks, model weights, dataset files etc.

It is recommended to use [this guide](readme_stuff/GoogleDriveFolderGuide.md) for the google drive folder, in order to understand the purpose of each file.

Each notebook starts with cloning this repository and makes use of the code present here.

### Prerequisites

It is recommended to use conda or venv environments if you're going to run this on your local PC.

### Installation

There's a requirement.txt file with the packages required to run this project. Although it is worth noting that my local machine didn't have a GPU (I used colab), so if you have a GPU you might need to install some additional packages (If you're willing to help, please do contribute the required changes for GPU compatibility).


<!-- USAGE EXAMPLES -->
## Dataset

[LHC Olympics R and D Anomaly Detection]() dataset was used. Although I made some modifications to it. In order to access the variant used in this repository please go to the [google drive folder](https://drive.google.com/drive/u/1/folders/1OkIPaDb25JooMULL0U5uzfqHfr2ZamDw).



<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/Ehsan1997/NormalizingFlows_HEP/issues) for a list of proposed features (and known issues).



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

Muhammad Ehsan ul Haq - [@EhsanBinEjaz](https://twitter.com/Ehsanbinejaz) - ehsanulhaq18@gmail.com

Project Link: [https://github.com/Ehsan1997/NormalizingFlows_HEP](https://github.com/Ehsan1997/NormalizingFlows_HEP)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Img Shields](https://shields.io)
* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Pages](https://pages.github.com)
* [Animate.css](https://daneden.github.io/animate.css)
* [Loaders.css](https://connoratherton.com/loaders)
* [Slick Carousel](https://kenwheeler.github.io/slick)
* [Smooth Scroll](https://github.com/cferdinandi/smooth-scroll)
* [Sticky Kit](http://leafo.net/sticky-kit)
* [JVectorMap](http://jvectormap.com)
* [Font Awesome](https://fontawesome.com)
* [Smash Icons](https://www.flaticon.com/authors/smashicons)
* [Flat Icons](https://www.flaticon.com)

<!-- SPECIAL THANKS -->
## Special Thanks
* [Sergei Glyzer](http://sergeigleyzer.com/)
* [Meenakshi Narain](https://twitter.com/meenakshinarain)
* [Emanule Usai](https://emanueleusai.com/)
* [Aneesh Heintz]()
* [Sinan Gençoğlu](https://www.linkedin.com/in/sinan-gencoglu/)



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/Ehsan1997/NormalizingFlows_HEP.svg?style=flat-square
[contributors-url]: https://github.com/Ehsan1997/NormalizingFlows_HEP/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Ehsan1997/NormalizingFlows_HEP.svg?style=flat-square
[forks-url]: https://github.com/Ehsan1997/NormalizingFlows_HEP/network/members
[stars-shield]: https://img.shields.io/github/stars/Ehsan1997/NormalizingFlows_HEP.svg?style=flat-square
[stars-url]: https://github.com/Ehsan1997/NormalizingFlows_HEP/stargazers
[issues-shield]: https://img.shields.io/github/issues/Ehsan1997/NormalizingFlows_HEP.svg?style=flat-square
[issues-url]: https://github.com/Ehsan1997/NormalizingFlows_HEP/issues
[license-shield]: https://img.shields.io/github/license/Ehsan1997/NormalizingFlows_HEP.svg?style=flat-square
[license-url]: https://github.com/Ehsan1997/NormalizingFlows_HEP/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/ehsansonofejaz
[product-screenshot]: readme_stuff/cc-image-lhc.jpg