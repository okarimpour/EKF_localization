#ifndef ObjectTracking
#define ObjectTracking

#include "ObjectTrackingConfig.h.in"
#include "Eigen/Dense"

#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <vector>
#include <math.h>
#include <utility> // std::pair
#include <stdexcept> // std::runtime_error
#include <sstream> // std::stringstream

class KalmanFilter {
    public:
        KalmanFilter();
        virtual ~KalmanFilter();

        void Initialize(Eigen::VectorXd &X_state_initial, Eigen::MatrixXd &P_Uncertainty_initial, Eigen::MatrixXd &F_state_transition_initial, 
                        Eigen::MatrixXd &H_measurement_function_initial, Eigen::MatrixXd &R_measurement_noise_initial,
                        Eigen::MatrixXd &Q_process_noise_initial);
        Eigen::MatrixXd CalculateJacobian(const Eigen::VectorXd &X_state);
        void Predict();
        void Update(const Eigen::VectorXd &Z_measurement);
        void UpdateEKF(const Eigen::VectorXd &Z_measurement);

        Eigen::VectorXd X_state;
        Eigen::MatrixXd P_Uncertainty; //Uncertainty Covarience Matrix (Square of Standard Devietion of noise)
        Eigen::MatrixXd F_state_transition;
        Eigen::MatrixXd H_measurement_function;
        Eigen::MatrixXd R_measurement_noise;
        Eigen::MatrixXd Q_process_noise;
};

class Measurement {
    public:
        enum SensorType{
            LIDAR,
            RADAR
        } sensor_type;

        double timestamp;
        Eigen::VectorXd raw_measurement;
};

class CSV {
	public:
    	void read_write_csv(std::string filename);
    private:
        Measurement measurement;
};

class SensorFusion {
    public:
        SensorFusion();
        virtual ~SensorFusion();
        Eigen::VectorXd ProcessMeasurement(const Measurement &measurement);
        KalmanFilter EKF;
    private:
        bool initialized;
        double previous_timestamp;
        double timestamp;
        //Eigen::VectorXd raw_measurement;
        Eigen::MatrixXd R_measurement_noise_lidar;
        Eigen::MatrixXd R_measurement_noise_radar;
        Eigen::MatrixXd H_measurement_function_lidar;
        Eigen::MatrixXd H_measurement_function_jacobian;
};

#endif
