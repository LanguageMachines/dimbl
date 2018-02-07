/*
  Copyright (c) 2006 - 2018
  CLST  - Radboud University
  ILK   - Tilburg University
  CLiPS - University of Antwerp

  This file is part of dimbl

  dimbl is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 3 of the License, or
  (at your option) any later version.

  dimbl is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, see <http://www.gnu.org/licenses/>.

  For questions and suggestions, see:
      https://github.com/LanguageMachines/dimbl/issues
  or send mail to:
      lamasoftware (at ) science.ru.nl
*/

#include <fstream>
#include <vector>
#include <cstdlib>
#include <sys/time.h>
#include <config.h>
#ifdef HAVE_OPENMP
#include <omp.h>
#endif
#include "timbl/TimblAPI.h"
#include "dimbl/DimProcs.h"

using namespace std;
using namespace Timbl;
using namespace TiCC;

//#define DEBUG 1

class mp_worker: public worker {
public:
  bool Init( TimblAPI *, const string&, const string&, int );
  bool writeTree( const string& );
  bool readTree( const TiCC::CL_Options&, const string&, const string&, int );
  bool Execute( const string & );
};

bool mp_worker::Init( TimblAPI *pnt,
		      const string& fileName,
		      const string& wfName, int i ){
  exp = pnt;
  cnt = i;
#ifdef HAVE_OPENMP
#if DEBUG > 0
#pragma omp critical
  cerr << "starting worker[" << cnt << "], thread = "
       << omp_get_thread_num()+1 << "/" << omp_get_num_threads() << endl;
#endif
#endif
  bool ok = exp->Learn( fileName );
  if ( !ok ){
    cerr << "learning failed for " << fileName << endl;
    cerr << "stopped" << endl;
    return false;
  }
  if ( !wfName.empty() )
    ok = exp->GetWeights( wfName, exp->CurrentWeighting() );
  if ( !ok ){
    cerr << "reading weights from " << wfName << " failed " << endl;
    cerr << "stopped" << endl;
    return false;
  }
  return true;
}

bool mp_worker::Execute( const string& line ){
  result = 0;
  if ( !line.empty() ){
#ifdef HAVE_OPENMP
#if DEBUG > 0
#pragma omp critical
    cerr << "exp-" << cnt << " thread = "
	 << omp_get_thread_num()+1 << "/" << omp_get_num_threads()
 	 << " classify '" << line << "'" << endl;
#endif
#endif
    result = exp->classifyNS( line );
    result->setShowDistance( true );
    result->setShowDistribution( true );
  }
  return true;
}

bool mp_worker::readTree( const TiCC::CL_Options& opts,
			  const string& name,
			  const string& wfName,
			  int i ){
  bool ok = false;
  exp = new TimblAPI( opts );
  cnt = i;
#ifdef HAVE_OPENMP
#if DEBUG > 0
#pragma omp critical
  cerr << "starting worker[" << cnt << "], thread = "
       << omp_get_thread_num()+1 << "/" << omp_get_num_threads() << endl;
#endif
#endif
  if ( exp->GetInstanceBase( name ) ){
    ok = exp->GetWeights( wfName, exp->CurrentWeighting() );
    if ( !ok ){
      cerr << "reading weights from " << wfName << " failed " << endl;
      cerr << "stopped" << endl;
    }
  }
  time_stamp( cout, "initialize a tree from " + name );
  return ok;
}

bool mp_worker::writeTree( const string& name ){
  return exp->WriteInstanceBase( name );
}

template <>
bool experiment<mp_worker>::createWorkers(){
  bool ok = true;
#ifdef HAVE_OPENMP
  if ( size < 1 )
    size = omp_get_max_threads();
  else {
    omp_set_num_threads( size );
    int mt = omp_get_max_threads();
    if ( mt < size ){
      cout << "couldn't get more then " << mt << " threads (requested was "
	   << size << " )" << endl;
      size = mt;
    }
  }
#else
  cerr << "couldn't get more then 1 thread. No OpenMP support. " << endl;
  size = 1;
#endif
  children.resize( size );
  split( config.trainFileName, size, config.tmpdir );
  time_stamp( cout, "   start training "+ toString(size) + " children:" );
  int i;
#pragma omp parallel private( i )
  {
    for( i=1; i <= size; ++i ){
#pragma omp single nowait
      {
	string iname = config.tmpdir
	  + TiCC::basename(config.trainFileName) + "-" + toString( i );
#if DEBUG > 0
	cerr << "start " << iname << endl;
#endif
	TimblAPI *child = new TimblAPI( *train );
	children[i-1].Init( child, iname, config.weightsFileName, i );
	cout << "child " << i << ":";
	children[i-1].exp->ShowIBInfo( cout );
      }
    }
  }
  time_stamp( cout, "finished training children:" );
  return ok;
}

template <>
bool experiment<mp_worker>::createWorkerFiles(){
#ifdef HAVE_OPENMP
  if ( size < 1 )
    size = omp_get_max_threads();
  else {
    omp_set_num_threads( size );
    int mt = omp_get_max_threads();
    if ( mt < size ){
      cout << "couldn't get more then " << mt << " threads (requested was "
	   << size << " )" << endl;
      size = mt;
    }
  }
#else
  cerr << "couldn't get more then 1 thread. No OpenMP support. " << endl;
  size = 1;
#endif
  children.resize( size );
  split( config.trainFileName, size, config.tmpdir );
  time_stamp( cout, "   start training "+ toString(size) + " children:" );
  ofstream os( config.treeOutFileName );
  int i;
  vector<bool> ok(size,true);
#pragma omp parallel private( i )
  {
    for( i=1; i <= size; ++i ){
#pragma omp single nowait
      {
	string iname = config.tmpdir
	  + TiCC::basename(config.trainFileName) + "-" + toString( i );
#if DEBUG > 0
	cerr << "start " << iname << endl;
#endif
	TimblAPI *child = new TimblAPI( *train );
	children[i-1].Init( child, iname, config.weightsFileName, i );
	string oname = config.tmpdir
	  + TiCC::basename(config.trainFileName) + "-" + toString( i ) + ".tree";
	os << oname << endl;
#if DEBUG > 0
	cerr << "start saving in" << oname << endl;
#endif
	if ( !children[i-1].writeTree( oname ) )
	  ok[i-1] = false;
	else {
#pragma omp critical
	  {
	    cout << "child " << i << ":";
	    children[i-1].exp->ShowIBInfo( cout );
	  }
	}
      }
    }
  }
  bool result = true;
  for( i=0; i < size; ++i ){
    result &= ok[i];
  }
  if ( result ){
    time_stamp( cout, "finished saving trees" );
    string wF =  config.weightsFileName;
    if ( wF.empty() )
      wF = config.treeOutFileName + ".wghts";
    result = train->SaveWeights( wF );
    if ( !result ){
      cerr << "saving weights in " << wF << " failed " << endl;
      cerr << "stopped" << endl;
    }
    else {
      cerr << "weights saved in " << wF << endl;
      os << wF << endl;
      time_stamp( cout, "filenames stored in: " + config.treeOutFileName );
    }
  }
  else
    time_stamp( cout, "failed saving trees" );
  return result;
}

template <>
bool experiment<mp_worker>::createWorkersFromFile( const TiCC::CL_Options& opts ){
  ifstream is( config.treeInFileName );
  if ( !is ){
    cerr << "unable to read file: " << config.treeInFileName << endl;
    return false;
  }
  vector<string> filenames;
  string line;
  while ( getline( is, line ) ){
    ifstream tmp( line );
    if ( !tmp ){
      cerr << "unable to read file: "<< line << endl;
      return false;
    }
    filenames.push_back( line );
  }
  size = filenames.size() - 1;
  string wFile;
  if ( size == 0 ){
    cerr << "no usefull info in" <<  config.treeInFileName << endl;
    return false;
  }
  else {
    wFile = filenames[size];
    ifstream tmp( wFile );
    if ( !tmp ){
      cerr << "unable to read weightsfile: " << wFile << endl;
      return false;
    }
    else
      cerr << "using weightsfile: " << wFile << endl;
  }
#ifdef HAVE_OPENMP
  omp_set_num_threads( size );
  int mt = omp_get_max_threads();
  if ( mt < size ){
    cerr << "couldn't get more then " << mt << " threads (requested was "
	 << size << " )" << endl;
    return false;
  }
#else
  cerr << "couldn't set to more then 1 thread. No OpenMP support. " << endl;
  size = 1;
#endif
  children.resize( size );
  time_stamp( cout, "   start reading input for "+ toString(size) + " children:" );
  int i;
  vector<bool> ok(size,true);
#pragma omp parallel private( i )
  {
    for( i=1; i <= size; ++i ){
#pragma omp single nowait
      {
	string iname = filenames[i-1];
#if DEBUG > 0
	cerr << "start " << iname << endl;
#endif
	if ( !children[i-1].readTree( opts, iname, wFile, i ))
	  ok[i-1] = false;
      }
    }
  }
  bool result = true;
  for( i=0; i < size; ++i ){
    result &= ok[i];
  }
  if ( result ){
    time_stamp( cout, "finished initializing children:" );
  }
  return result;
}

void usage(){
  cerr << "usage:" << "\tdimbl -S threads [Timbl Options]" << endl
       << "\twhere 'threads' is a number specifying the number of" << endl
       << "\tthreads that is used for training" << endl
       << "\t'Timbl Options' specifies the Timbl Options to use." << endl
       << "\tNote that not all Timbl options are working yet" << endl
       << "\tmost usefull are: -f, -t, -i " << endl;
}

int main(int argc, char *argv[]) {
  cerr << "dimbl " << VERSION << " (c) CLST/ILK 1998 - 2018" << endl;
  cerr << "Centre for Language and Speech Technology, Radboud University" << endl;
  cerr << "Induction of Linguistic Knowledge Research Group, Tilburg University" << endl;
  cerr << "based on [" << Timbl::VersionName() << "]" << endl;
  if ( argc < 2 ){
    usage();
    return EXIT_SUCCESS;
  }
  try {
    TiCC::CL_Options opts;
    opts.init( argc, argv ); // we don't check the arguments for now
    string value;
    if ( opts.is_present( 'h' ) ){
      usage();
      return EXIT_SUCCESS;
    }
    if ( opts.is_present( 'V' ) || opts.is_present( "version" ) ){
      return EXIT_SUCCESS;
    }
    experiment<mp_worker> theExp( opts );
    if ( !theExp.config.treeInFileName.empty() ){
      if ( !theExp.config.treeOutFileName.empty() ){
	cerr << "both -i and -I specified! That is impossible to handle"
	     << endl;
	return EXIT_FAILURE;
      }
      if ( !theExp.createWorkersFromFile( opts ) )
	return EXIT_FAILURE;
    }
    else {
      theExp.Train( opts );
      if ( !theExp.config.treeOutFileName.empty() ){
	theExp.createWorkerFiles();
	cout << "done." << endl;
	return EXIT_SUCCESS;
      }
      else
	if ( !theExp.createWorkers() )
	  return EXIT_FAILURE;
    }
    int lineCount = 0;
    time_t startTime;
    time(&startTime);
    timeval Start;
    gettimeofday( &Start, 0 );
    string line;
    theExp.initStatistics();
    time_stamp( cout, "start testing '" + theExp.config.testFileName + "'" );
    while ( getline( theExp.config.inp, line ) ){
      ++lineCount;
      if ( line.empty() )
	continue;
      int i;
#pragma omp parallel for private(i)
      for ( i=0; i < theExp.size; ++i ){
	theExp.children[i].Execute( line );
      }
      neighborSet result = theExp.Finalize();
      theExp.showResult( result, line );
      theExp.showProgress( cout, lineCount, startTime );
    }
    time_stamp( cout, "Ready:  ", lineCount );
    theExp.showStatistics( cout );
    show_speed_summary( cout, lineCount, Start );
    cout << "results in " << theExp.config.outFileName << endl;
  }
  catch ( exception& e ){
    cerr << "terminated because of " << e.what() << endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
