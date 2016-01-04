/*
  Copyright (c) 2006 - 2016
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
#ifndef DIMPROCS_H
#define DIMPROCS_H

#include <cstring>
#include <iomanip>
#include <unistd.h> // for unlink()
#include "ticcutils/StringOps.h"
#include "timbl/Statistics.h"

class settings {
 public:
  settings( TiCC::CL_Options& );
  ~settings(){ delete distanceMetric; };
  bool ok() const { return status; };
  int numThreads;
  std::string trainFileName;
  std::string testFileName;
  std::string weightsFileName;
  std::string treeOutFileName;
  std::string treeInFileName;
  std::string outFileName;
  std::string tmpdir;
  bool wFileSpecified;
  bool keepfiles;
  mutable std::ifstream inp;
  mutable std::ofstream out;
  Timbl::decayStruct *distanceMetric;
  Timbl::InputFormatType IF;
  int nn;
  bool do_distance;
  bool do_distrib;
  bool do_neighb;
  bool do_confusion;
  bool do_class_stats;
  bool do_advanced_stats;
  int beam;
  int progress;
  int estimate;
 private:
  bool status;
};

class worker{
public:
  worker(){ exp = 0; cnt = 0; };
  virtual ~worker();
  virtual bool Init( Timbl::TimblAPI *,
		     const std::string&, const std::string&, int ) = 0;
  virtual bool Execute( const std::string & ) = 0;
  Timbl::TimblAPI *exp;
  int cnt;
  const Timbl::neighborSet *Result() { return result; };
protected:
  const Timbl::neighborSet *result;
};

void split( const std::string& , int, const std::string& );
void time_stamp( std::ostream&, const std::string&, int = -1 );


template <class working>
class experiment {
 public:
 experiment( TiCC::CL_Options& opts ): config( opts ){
    if ( !config.ok() ){
      throw std::runtime_error( "error in config" );
    }
    size = config.numThreads;
    train =0;
    confusion=0;
    targets = 0;
  }
  virtual ~experiment();
  bool Train( const TiCC::CL_Options& );
  virtual bool createWorkers();
  virtual bool createWorkersFromFile( const TiCC::CL_Options& );
  virtual bool createWorkerFiles();
  Timbl::neighborSet Finalize();
  void showResult( Timbl::neighborSet&, const std::string& );
  void showProgress( std::ostream&, int, time_t );
  void initStatistics();
  void showStatistics( std::ostream& ) const;
  int size;
  settings config;
  std::vector<working> children;
  Timbl::TimblAPI *train;
  Timbl::StatisticsClass stats;
  Timbl::ConfusionMatrix *confusion;
  const Timbl::Target *targets;
};

template <class working>
experiment<working>::~experiment() {
  if ( !config.keepfiles
       && config.treeInFileName.empty() ){
    std::cout << "cleanup intermediate files in " << config.tmpdir << std::endl;
    for( int i=0; i < size; ++i ){
      std::string iname = config.tmpdir
	+ TiCC::basename(config.trainFileName) + "-" + TiCC::toString( i+1 );
      unlink( iname.c_str() );
    }
  }
  children.clear();
  delete train;
}

template <class working>
bool experiment<working>::Train( const TiCC::CL_Options& opts ){
  bool ok = true;
  train = new Timbl::TimblAPI( opts, "train" );
  if ( !train ){
    throw std::runtime_error( "training failed" );
  }
  time_stamp( std::cout,
	      "learning using '"
	      + TiCC::basename(config.trainFileName)
	      + "':" );
  ok = train->Learn( config.trainFileName );
  if ( !config.weightsFileName.empty() ){
    //     std::cout << "saving weights in '"
    // 	      << config.weightsFileName << "'" << std::endl;
    ok = train->SaveWeights( config.weightsFileName );
  }
  if ( ok ){
    train->initExperiment();
  }
  return ok;
}

template <class working>
Timbl::neighborSet experiment<working>::Finalize(){
  Timbl::neighborSet result;
  result.setShowDistance( config.do_distance );
  result.setShowDistribution( config.do_distrib );
  for ( int i=0; i < size; ++i ){
    const Timbl::neighborSet *ns = children[i].Result();
    if ( ns ) {// if mismatch or the like, no result
      result.merge( *ns );
    }
  }
  result.truncate( config.nn  );
  return result;
}

template <class working>
void experiment<working>::showResult( Timbl::neighborSet& res,
				      const std::string& orig ){
  if ( res.size() == 0 ){
    config.out << orig << " FAILED TEST" << std::endl;
  }
  else {
    const Timbl::ValueDistribution *db
      = res.bestDistribution( config.distanceMetric, config.nn - 1  );
    bool Tie;
    const Timbl::TargetValue *tv = db->BestTarget( Tie );
    if ( Tie ){
      const Timbl::ValueDistribution *db2
	= res.bestDistribution( config.distanceMetric, config.nn  );
      const Timbl::TargetValue *tv2 = db2->BestTarget( Tie );
      if ( !Tie ){
	delete db;
	db = db2;
	tv = tv2;
      }
      else {
	delete db2;
	res.truncate( config.nn - 1 );
      }
    }
    else
      res.truncate( config.nn - 1 );
    config.out << orig;
    switch ( config.IF ){
    case Timbl::Columns:
      config.out << " ";
      break;
    case Timbl::Compact:
      break;
    default:
      config.out << ",";
      break;
    }
    config.out << tv;
    if ( config.do_distance || config.do_distrib ){
      if ( config.do_distrib )
	config.out << " " << db->DistToStringW( config.beam );
      if ( config.do_distance ){
	int OldPrec = config.out.precision(DBL_DIG-1);
	config.out.setf(std::ios::showpoint);
	config.out.width(8);
	config.out << " " << res.bestDistance();
	config.out.precision(OldPrec);
      }
    }
    config.out << std::endl;
    if ( config.do_neighb ){
      config.out << res;
    }
    /////
    bool exact = fabs( res.bestDistance() ) < DBL_EPSILON ;
    stats.addLine();
    if ( exact )
      stats.addExact();
    Timbl::TargetValue *iTV = children[0].exp->lastHandledInstance()->TV;
    if ( confusion )
      confusion->Increment( iTV, tv );
    bool correct = iTV && ( tv->Index() == iTV->Index() );
    if ( correct ){
      stats.addCorrect();
      if ( Tie )
	stats.addTieCorrect();
    }
    else if ( Tie )
      stats.addTieFailure();
    /////
    delete db;
  }
}

template <class working>
void experiment<working>::initStatistics(){
  if ( config.do_advanced_stats ){
    targets = children[0].exp->myTargets();
    confusion = new Timbl::ConfusionMatrix( targets->ValuesArray.size() );
  }
}

template <class working>
void experiment<working>::showStatistics( std::ostream& os ) const {
  os << std::endl;
  if ( confusion )
    confusion->FScore( os, targets, config.do_class_stats );
  os << "overall accuracy:        "
     << stats.testedCorrect()/(double) stats.dataLines()
     << "  (" << stats.testedCorrect() << "/" << stats.dataLines()  << ")" ;
  if ( stats.exactMatches() != 0 )
    os << ", of which " << stats.exactMatches() << " exact matches " ;
  os << std::endl;
  int totalTies =  stats.tiedCorrect() + stats.tiedFailure();
  if ( totalTies > 0 ){
    if ( totalTies == 1 )
      os << "There was 1 tie";
    else
      os << "There were " << totalTies << " ties";
    double tie_perc = 100 * ( stats.tiedCorrect() / (double)totalTies);
    int oldPrec = os.precision(2);
    os << " of which " << stats.tiedCorrect()
       << " (" << std::setprecision(2)
       << tie_perc << std::setprecision(6) << "%)";
    if ( totalTies == 1 )
      os << " was correctly resolved" << std::endl;
    else
      os << " were correctly resolved" << std::endl;
    os.precision(oldPrec);
  }
  if ( confusion && config.do_confusion ){
    os << std::endl;
    confusion->Print( os, targets );
  }
}

template <class working>
void experiment<working>::showProgress( std::ostream& os,
					int line, time_t start ){
  char time_string[26];
  struct tm *curtime;
  time_t Time;
  time_t SecsUsed;
  time_t EstimatedTime;
  double Estimated;
  int local_progress = config.progress;

  if ( ( (line % local_progress ) == 0) || ( line <= 10 ) ||
	 ( line == 100 || line == 1000 || line == 10000 ) ){
    time(&Time);
    if ( line == 1000 ){
      // check if we are slow, if so, change progress value
      if ( Time - start > 120 ) // more then two minutes
	// very slow !
	local_progress = 1000;
    }
    else if ( line == 10000 ){
      if ( Time - start > 600 ) // more then ten minutes
	// quit slow !
	local_progress = 10000;
    }
    curtime = localtime(&Time);
    os << "Tested: ";
    os.width(6);
    os.setf(std::ios::right, std::ios::adjustfield);
    strcpy( time_string, asctime(curtime));
    time_string[24] = '\0';
    os << line << " @ " << time_string;

    // Estime time until Estimate.
    //
    if ( config.estimate > 0 ) {
      SecsUsed = Time - start;
      if ( SecsUsed > 0 ) {
	Estimated = (SecsUsed / (float)line) *
	  (float)config.estimate;
	EstimatedTime = (long)Estimated + start;
	os << ", ";
	strcpy(time_string, ctime(&EstimatedTime));
	time_string[24] = '\0';
	os << config.estimate << ": " << time_string;
      }
    }
    os << std::endl;
  }
}

inline std::string curTime();
void show_speed_summary( std::ostream&, int, const timeval& );

#endif
