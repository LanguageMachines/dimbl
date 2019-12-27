/*
  Copyright (c) 2006 - 2020
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

#include <ostream>
#include <iomanip>
#include <fstream>
#include <sys/time.h>

#include "timbl/TimblAPI.h"

using namespace std;
using namespace Timbl;
using namespace TiCC;

#include "dimbl/DimProcs.h"

worker::~worker(){
  if ( exp )
    delete exp;
}

void split( const string& file, int num, const string& tmpdir ){
  int cnt=0;
  string line;
  ifstream is( file );
  while ( getline( is, line ) )
    ++cnt;
  if ( cnt ){
    is.clear();
    is.seekg(0);
    int to_do = cnt/num + 1;
    for ( int i=0; i < num; ++i ){
      string oname = tmpdir + TiCC::basename(file) + "-" + toString( i+1 );
      ofstream os( oname );
      if ( os ){
	int done = 0;
	while ( done < to_do &&
		getline( is, line ) ){
	  os << line << endl;
	  ++done;
	}
	if ( i == num-1 ){
	  while ( getline( is, line ) ){
	    os << line << endl;
	  }
	}
      }
    }
  }
}

settings::settings( TiCC::CL_Options& opts ){
  status = true;
  wFileSpecified=false;
  distanceMetric = 0;
  nn = 2;
  numThreads = 1;
  do_distance = false;
  do_distrib = false;
  do_neighb = false;
  do_confusion = false;
  do_class_stats = false;
  do_advanced_stats = false;
  beam = 0;
  progress = 10000;
  estimate = 0;
  IF = UnknownInputFormat;
  tmpdir = "/tmp/";
  keepfiles = false;
  string value;
  if ( opts.is_present( 'a', value ) ){
    Algorithm tst;
    if ( !string_to(value,tst) || tst != IB1 ){
      cerr << "unsupported algorithm. only IB1 is possible!" << endl;
      status = false;
    }
  }
  if ( opts.extract( 'S', value ) ){
    numThreads = stringTo<int>( value );
  }
  if ( opts.extract( 'k', value ) ){
    nn = stringTo<int>(value) + 1;
  }
  opts.insert( 'k', toString<int>( nn ), true );
  if ( opts.extract( "Beam", value ) ){
    beam = stringTo<int>(value);
  }
  if ( opts.extract( "tempdir", value ) ){
    tmpdir = value;
    if ( !tmpdir.empty() && tmpdir[tmpdir.length()-1] != '/' ){
      tmpdir += "/";
    }
  }
  if ( opts.extract( "keep", value ) ){
    keepfiles = true;
  }
  if ( opts.extract( 'e', value ) ){
    estimate = stringTo<int>(value);
  }
  if ( opts.is_present( 'F', value ) ){
    if ( !stringTo<InputFormatType>( value, IF ) ){
      cerr << "illegal value for -F option: " << value << endl;
      status = false;
    }
  }
  if ( opts.extract( 'p', value ) ){
    progress = stringTo<int>(value);
  }
  if ( opts.extract( 'f', value ) ){
    trainFileName = value;
  }
  if ( opts.extract( 't', value ) ){
    testFileName = value;
  }
  if ( opts.extract( 'i', value ) ){
    treeInFileName = value;
  }
  if ( opts.extract( 'I', value ) ){
    treeOutFileName = value;
  }
  if ( opts.extract( 'o', value ) ){
    outFileName = value;
  }
  else
    outFileName = testFileName + ".out";
  if ( trainFileName.empty() && treeInFileName.empty() ){
    cerr << "missing trainfile or -i option" << endl;
    status = false;
  }
  else if ( testFileName.empty() && treeOutFileName.empty() ){
    cerr << "missing testfile" << endl;
    status = false;
  }
  else {
    Weighting W = GR;
    if ( opts.is_present( 'w', value ) ){
      // user specified weighting
      if ( !string_to( value, W ) ){
	// No valid weighting, so assume it also has a filename
	vector<string> parts;
	size_t num = TiCC::split_at( value, parts, ":" );
	if ( num == 2 ){
	  if ( !string_to( parts[1], W ) ){
	    cerr << "invalid weighting option: " << value << endl;
	    status = false;
	  }
	  if ( status ){
	    weightsFileName = parts[0];
	    wFileSpecified = true;
	    opts.remove( 'w' );
	  }
	}
	else if ( num == 1 ){
	  weightsFileName = value;
	  wFileSpecified = true;
	  W = GR;
	  opts.remove( 'w' );
	}
	else {
	  cerr << "invalid weighting option: " << value << endl;
	  status = false;
	}
      }
      else {
	// valid Weight, but maybe a number, so replace anyway
	opts.remove( 'w' );
      }
      if ( trainFileName.empty() && !weightsFileName.empty() ){
	cerr << "a weights filename is specied. "
	     << "That is incompatible with the -i option" << endl;
	status = false;
      }
      opts.insert( 'w', to_string(W), false );
    }
  }
  if ( status ) {
    if ( opts.extract( 'd', value ) ){
      // user specified distance metric
      DecayType decay = UnknownDecay;
      double decay_alfa = 1.0;
      double decay_beta = 1.0;
      string::size_type pos1 = value.find( ":" );
      if ( pos1 == string::npos ){
	pos1 = value.find_first_of( "0123456789" );
	if ( pos1 != string::npos ){
	  if ( ! ( stringTo<DecayType>( string( value, 0, pos1 ), decay ) &&
		   stringTo<double>( string( value, pos1 ), decay_alfa ) ) ){
	    cerr << "illegal value for -d option: " << value << endl;
	    status = false;
	  }
	}
	else if ( !stringTo<DecayType>( value, decay ) ){
	  cerr << "illegal value for -d option: " << value << endl;
	  status = false;
	}
      }
      else {
	string::size_type pos2 = value.find( ':', pos1+1 );
	if ( pos2 == string::npos ){
	  pos2 = value.find_first_of( "0123456789", pos1+1 );
	  if ( pos2 != string::npos ){
	    if ( ! ( stringTo<DecayType>( string( value, 0, pos1 ),
					  decay ) &&
		     stringTo<double>( string( value, pos2 ),
				       decay_alfa ) ) ){
	      cerr << "illegal value for -d option: " << value << endl;
	      status = false;
	    }
	  }
	  else {
	    cerr << "illegal value for -d option: " << value << endl;
	    status = false;
	  }
	}
	else {
	  if ( ! ( stringTo<DecayType>( string( value, 0, pos1 ), decay ) &&
		   stringTo<double>( string( value, pos1+1, pos2-pos1-1 ),
				     decay_alfa ) &&
		   stringTo<double>( string( value, pos2+1 ),
				     decay_beta ) ) ){
	    cerr << "illegal value for -d option: " << value << endl;
	    status = false;
	  }
	}
      }
      if ( status ){
	switch ( decay  ){
	case Zero:
	  distanceMetric = new zeroDecay;
	  break;
	case InvDist:
	  distanceMetric = new invDistDecay;
	  break;
	case InvLinear:
	  distanceMetric = new invLinDecay;
	  break;
	case ExpDecay:
	  distanceMetric = new expDecay( decay_alfa, decay_beta );
	  break;
	default:
	  cerr << "ignoring unknown decay" << toString( decay ) << endl;
	}
      }
    }
  }
  if ( status ){
    bool mood;
    if ( opts.extract( 'v', value, mood ) ){
      if ( value.find( "di" ) != string::npos )
	do_distance = true;
      if ( value.find( "db" ) != string::npos )
	do_distrib = true;
      if ( value.find( "k" ) != string::npos )
	do_neighb = true;
      if ( value.find( "cm" ) != string::npos ){
	do_confusion = true;
	do_advanced_stats = true;
      }
      if ( value.find( "cs" ) != string::npos ){
	do_class_stats = true;
	do_advanced_stats = true;
      }
      if ( value.find( "as" ) != string::npos )
	do_advanced_stats = true;
    }
    opts.insert( 'v', "S", true );
    if ( treeOutFileName.empty() ){
      inp.open( testFileName );
      if ( !inp ){
	cerr << "unable to open " << testFileName << endl;
	status = false;
      }
    }
    out.open( outFileName );
    if ( !out ){
      cerr << "Unable to open outputfle " << outFileName << endl;
      status = false;
    }
  }
}

inline string curTime(){
  time_t lTime;
  struct tm *curtime;
  char time_buf[64];
  time(&lTime);
  curtime = localtime(&lTime);
  strftime( time_buf, sizeof(time_buf), "%a %b %e %T %Y", curtime );
  return time_buf;
}

void time_stamp( ostream& os, const string& line, int number ) {
  os << line;
  if ( number > -1 ){
    os.width(6);
    os.setf(ios::right, ios::adjustfield);
    os << number << " @ ";
  }
  else
    os << "        ";
  os << curTime() << endl;
}

void show_speed_summary( ostream& os, int lines,
			 const timeval& Start ) {
  timeval Time;
  gettimeofday( &Time, 0 );
  long int uSecsUsed = (Time.tv_sec - Start.tv_sec) * 1000000 +
    (Time.tv_usec - Start.tv_usec);
  double secsUsed = (double)uSecsUsed / 1000000 + DBL_EPSILON;
  int oldPrec = os.precision(4);
  os << setprecision(4);
  os.setf( ios::fixed, ios::floatfield );
  os << "Seconds taken: " << secsUsed << " (";
  os << setprecision(2);
  os << lines / secsUsed << " p/s)" << endl;
  os << setprecision(oldPrec);
}
