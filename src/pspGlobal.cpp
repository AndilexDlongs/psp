// Copyright 2022 <Lenard Dome> [legal/copyright]
// [[Rcpp::depends(RcppArmadillo)]]
#include <mpi.h>
#include <RcppArmadillo.h>
#include <chrono>
#include <fstream>

// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

// utility functions
// Weisstein, Eric W. "Hypersphere Point Picking." From MathWorld.
// https://mathworld.wolfram.com/HyperspherePointPicking.html
// pick new jumping distributions from the unit hypersphere scaled by the radius
mat HyperPoints(int counts, int dimensions, double radius) {
  // create a uniform distribution
  mat hypersphere = randn(counts, dimensions, distr_param(0, 1) );
  colvec denominator = sum(square(hypersphere), 1);
  denominator = 1 / sqrt(denominator);
  // pick points from within unite hypersphere
  hypersphere = hypersphere.each_col() % denominator;
  // scale up values by r
  rowvec rad = randu<rowvec>(dimensions, distr_param(0.0, radius));
  hypersphere = hypersphere.each_row() % rad;
  return(hypersphere);
}

// constrain new jumping distributions within given parameter bounds
mat ClampParameters(mat jumping_distribution, colvec lower, colvec upper) {
  for (int i = 0; i < upper.n_elem; i++) {
    jumping_distribution.col(i).clamp(lower[i], upper[i]);
  }
  return(jumping_distribution);
}

// returns the unique slices of a cube (a 3D array) 
uvec FindUniqueSlices(cube predictions) {
  vec predictions_filter(predictions.n_slices, fill::zeros);
  // filter the same predictions
  for (uword x = 0; x < predictions.n_slices; x++) {
    vec current = vectorise( predictions.slice(x) );
    for (uword y = x + 1; y < predictions.n_slices; y++) {
      vec base = vectorise( predictions.slice(y) );
      uvec result = (base == current);
      if (all(result == 1)) predictions_filter(x) += 1;
    }
  }
  // remove multiple predictions for the comparisons
  uvec inclusion = find( predictions_filter == 0 );
  return(inclusion);
}

// compare two cubes of inequality matrices
// returns the complete list of unique ordinal matrices
cube OrdinalCompare(cube discovered, cube predicted) {

  cube drawer(discovered);
  mat index(predicted.n_slices, discovered.n_slices);

  // carry out the comparisons
  for (int x = 0; x < predicted.n_slices; x++) {
    mat current = predicted.slice(x);
    for (int y = 0; y < discovered.n_slices; y++) {
      mat base = discovered.slice(y);
      umat result = (base == current);
      index(x, y) =  any(vectorise(result) == 0);
    }
    if (all(index.row(x) == 1)) {
      cube update = join_slices(drawer, current);
      drawer = update;
    }
  }
  return(drawer);
}

// returns the last evaluated parameters in all chains in the MCMC
mat LastEvaluatedParameters(cube discovered, cube predicted, mat jumping, mat centers) {
  mat parameters(centers);
  mat index(discovered.n_slices, predicted.n_slices);
  // create index matrix
  for (uword x = 0; x < discovered.n_slices; x++) {
    mat current = discovered.slice(x);
    for (uword y = 0; y < predicted.n_slices; y++) {
      mat base = predicted.slice(y);
      umat result = (base == current);
      index(x, y) =  any(vectorise(result) == 0);
    }
    if (all(index.row(x))) {
      //  if there is a new region, appeng params to centers
      parameters.insert_rows(parameters.n_rows, jumping.rows( find(index.row(x) == 1, 1, "last") ));
    } else {
      // if there is an old region in predicted, update center
      parameters.row(x) = jumping.rows( find(index.row(x) == 0, 1, "last") );
    }
    // replace old centers with new ones
  }
  return(parameters);
}

// count ordinal patterns
rowvec CountOrdinal(cube updated_ordinal, cube predicted, rowvec counts) {
  mat index(updated_ordinal.n_slices, predicted.n_slices);
  rowvec new_counts = counts;
  new_counts.resize(updated_ordinal.n_slices);
  for (uword x = 0; x < updated_ordinal.n_slices; x++) {
    mat current = updated_ordinal.slice(x);
    for (uword y = 0; y < predicted.n_slices; y++) {
      mat base = predicted.slice(y);
      umat result = (base == current);
      if (all(vectorise(result) == 1)) {
        new_counts[x] += 1;
      }
    }
  }
  return(new_counts);
}

// match jumping distributions to ordinal ordinal_patterns
// returns a column uvec of slice IDs corresponding to each set in jumping_distribution
vec MatchJumpDists(cube updated_ordinal, cube predicted) {
  mat index(updated_ordinal.n_slices, predicted.n_slices);
  vec matches(predicted.n_slices);
  for (uword x = 0; x < updated_ordinal.n_slices; x++) {
    mat current = updated_ordinal.slice(x);
    for (uword y = 0; y < predicted.n_slices; y++) {
      mat base = predicted.slice(y);
      umat result = (base == current);
      if (all(vectorise(result) == 1)) {
        matches(y) = x;
      }
    }
  }
  return(matches + 1); // add one as c++ starts from 0
}

// create local csv file for storing coordinates
void CreateFile(CharacterVector names, std::string path_to_file) {
  std::ofstream outFile(path_to_file.c_str());
  outFile << "iteration,";
  for (uword i = 0; i < names.size(); i++) {
    outFile << names[i];
    outFile << + ",";
  }
  outFile << "pattern,\n";
}

// writes rows to csv file
void WriteFile(int iteration, mat evaluation, vec matches,
  std::string path_to_file) {
  // open file stream connection
  std::ofstream outFile(path_to_file.c_str(), std::ios::app);
  int rows = evaluation.n_rows;
  int columns = evaluation.n_cols;
  for (uword i = 0; i < rows; i++) {
    outFile << iteration;
    outFile << ",";
    for (uword k = 0; k < columns; k++) {
      outFile << evaluation(i, k);
      outFile << ",";
    }
    outFile << matches(i);
    outFile << ",\n";
  }
}

// [[Rcpp::export]]
List pspGlobal(Function model, Function discretize, List control, bool save, std::string path, std::string extension, bool quiet) {
  // Initialize MPI
  MPI_Init(NULL, NULL);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Log start time (only on root process)
  std::ofstream log_file;
  if (rank == 0) {
      std::string log_file_path = path + "_pspGlobal_log.txt";
      log_file.open(log_file_path, std::ios_base::app);
      auto start_total = std::chrono::high_resolution_clock::now();
      log_file << "[START] pspGlobal started\n";
  }

  // Import thresholds from control
  int max_iteration = as<int>(control["iterations"]);
  int population = as<int>(control["population"]);
  if (!max_iteration) max_iteration = std::numeric_limits<int>::max();
  if (!population) population = std::numeric_limits<int>::max();
  if (population == std::numeric_limits<int>::max() && max_iteration == std::numeric_limits<int>::max()) {
      if (rank == 0) Rcpp::stop("A reasonable threshold must be set by either adjusting iteration or population.");
      MPI_Finalize();
      return List::create();
  }

  double radius = as<double>(control["radius"]);
  mat init = as<mat>(control["init"]);
  colvec lower = as<colvec>(control["lower"]);
  colvec upper = as<colvec>(control["upper"]);
  int dimensions = init.n_cols;
  if (dimensions != lower.n_elem || dimensions != upper.n_elem) {
      if (rank == 0) Rcpp::stop("init, lower, and upper must have the same length.");
      MPI_Finalize();
      return List::create();
  }
  int dimensionality = as<int>(control["dimensionality"]);
  int response_length = as<int>(control["responses"]);

  // Seed setup
  Rcpp::Environment base_env("package:base");
  Rcpp::Function set_seed_r = base_env["set.seed"];

  // Prepare initial distributions and allocate arrays
  mat last_eval = init;
  mat jumping_distribution = init;
  mat continuous(jumping_distribution.n_rows, response_length);
  cube ordinal(dimensionality, dimensionality, jumping_distribution.n_rows);

  bool parameter_filled = false;
  int iteration = 0;
  uvec underpopulated = { 0 };

  // Iterate until the parameter space is filled or max iterations are reached
  while (!parameter_filled) {
      iteration++;

      // Divide work for MPI processes
      int num_points = init.n_rows;
      int points_per_process = num_points / size;
      int remainder = num_points % size;
      int start_index = rank * points_per_process + std::min(rank, remainder);
      int end_index = start_index + points_per_process + (rank < remainder ? 1 : 0);

      // Evaluate jumping distributions for each MPI process
      for (int i = start_index; i < end_index; i++) {
          NumericVector probabilities = model(jumping_distribution.row(i));
          NumericMatrix teatime = discretize(probabilities);
          rowvec responses = as<rowvec>(probabilities);
          continuous.row(i) = responses;
          mat evaluate = as<mat>(teatime);
          ordinal.slice(i) = evaluate;
      }

      // Gather results from all processes to root process
      if (rank == 0) {
          mat all_continuous(num_points, response_length);
          cube all_ordinal(dimensionality, dimensionality, num_points);

          std::vector<int> recvcounts(size);
          std::vector<int> displs(size);
          for (int i = 0; i < size; i++) {
              recvcounts[i] = (num_points / size) + (i < remainder ? 1 : 0);
              displs[i] = (i == 0) ? 0 : displs[i - 1] + recvcounts[i - 1];
          }

          MPI_Gatherv(continuous.memptr() + start_index * response_length, (end_index - start_index) * response_length, MPI_DOUBLE,
                      all_continuous.memptr(), recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

          // Process the gathered results on root
          uvec include = FindUniqueSlices(all_ordinal); // Assuming this function is defined elsewhere
          last_eval = LastEvaluatedParameters(storage, all_ordinal.slices(include),
                                              jumping_distribution.rows(include), last_eval); // Assuming this function is defined elsewhere
          storage = OrdinalCompare(storage, all_ordinal.slices(include)); // Assuming this function is defined elsewhere
          counts = CountOrdinal(storage, all_ordinal, counts); // Assuming this function is defined elsewhere

          underpopulated = find(counts < population);

          // Check termination condition
          if (iteration == max_iteration || underpopulated.n_elem == 0) {
              parameter_filled = true;
          }
      }

      // Broadcast termination condition to all processes
      MPI_Bcast(&parameter_filled, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
      if (parameter_filled) break;

      // Optional: Write data to disk
      if (save && rank == 0) {
          vec match = MatchJumpDists(storage, ordinal); // Assuming this function is defined elsewhere
          WriteFile(iteration, jumping_distribution, match, path + "_parameters" + extension); // Assuming this function is defined elsewhere
          WriteFile(iteration, continuous, match, path + "_continuous" + extension); // Assuming this function is defined elsewhere
      }

      // Logging iteration time
      if (rank == 0) {
          log_file << "[ITERATION] " << iteration << " completed.\n";
      }
  }

  // Log total time
  if (rank == 0) {
      auto end_total = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> duration_total = end_total - start_total;
      log_file << "[END] pspGlobal completed in " << duration_total.count() << " seconds\n";
      log_file.close();
  }

  // Finalize MPI
  MPI_Finalize();

  // Return results from rank 0
  if (rank == 0) {
      List out = Rcpp::List::create(
          Rcpp::Named("ordinal_patterns") = storage,
          Rcpp::Named("ordinal_counts") = counts,
          Rcpp::Named("iterations") = iteration
      );
      return out;
  }
  return List::create();
}
