[![GitHub build](https://github.com/LanguageMachines/dimbl/actions/workflows/dimbl.yml/badge.svg?branch=master)](https://github.com/LanguageMachines/dimbl/actions/)
[![Language Machines Badge](http://applejack.science.ru.nl/lamabadge.php/dimbl)](http://applejack.science.ru.nl/languagemachines/)
[![DOI](https://zenodo.org/badge/9028755.svg)](https://zenodo.org/badge/latestdoi/9028755)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)


Distributed Tilburg Memory Based Learner
=======================================

    Dimbl utils 0.17 (c) ILK/CLST 1998 - 2024
    by Ko van der Sloot

    Tilburg centre for Cognition and Communication, Tilburg University.
    Centre for Language and Speach Technology, Radboud University
    KNAW Humanities Cluster

Comments and bug-reports are welcome at our issue tracker at
https://github.com/LanguageMachines/dimbl/issues or by mailing
lamasoftware (at) science.ru.nl.
Updates and more info may be found on
https://languagemachines.github.io/timbl .

dimbl is distributed under the GNU Public Licence (see the file COPYING)


This software has been tested on:
- Intel platform running several versions of Linux, including Ubuntu and Debian
  both on 32 an 64 bit
- Windows platform using Cygwin
- APPLE running OSX 10.5

Compilers:
- GCC (7 - 11). The compiler MUST support OpenMP
- CLang (14-15) The compiler MUST support OpenMP

Contents of this distribution:
- sources
- Licensing information ( COPYING )
- Installation instructions ( INSTALL )
- Build system based om Gnu Autotools
- example data files ( in the demos directory )
- documentation ( in the docs directory )


Dependecies:
To be able to succesfully build Dimbl from the tarball, you need the
following pakages:
- pkg-config
- ticcutils
- timlb
