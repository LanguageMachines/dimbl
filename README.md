dimbl 0.16 (c) CLST/ILK/CLiPS 1998-2019

Distributed Tilburg Memory Based Learner

Centre for Language and Speech Technology, Radboud University,
Induction of Linguistic Knowledge Research Group, Tilburg University and
Centre for Dutch Language and Speech, University of Antwerp

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
- GCC (4.9 - 7). The compiler MUST support OpenMP
- CLang (4 - 7). The compiler MUST support OpenMP

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
