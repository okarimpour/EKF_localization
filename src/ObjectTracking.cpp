#include "ObjectTracking.h"

void CSV::read_write_csv(std::string filename) {
	SensorFusion sensorFusion;
	std::string splitString;
    std::ifstream myFile;
	myFile.open(filename);
	std::ofstream myNewFile("../data/output.csv");

	if(!myNewFile.is_open()) {
		throw std::runtime_error("Could not open output file");
	}
    if(!myFile.is_open()) {
		throw std::runtime_error("Could not open input file");
	}

    if(myFile.good()) {
		std::string sensor_type;
		double timestamp, position_x, position_y, radial_distance, angle, radial_velocity;
		while(std::getline(myFile, sensor_type, ',')) {
			if (sensor_type == "LIDAR") {
				measurement.sensor_type = Measurement::LIDAR;
				measurement.raw_measurement = Eigen::VectorXd(2);

				std::getline(myFile, splitString, ',');
				timestamp = std::stod(splitString);
				measurement.timestamp = timestamp;

				getline(myFile, splitString, ',');
				position_x = std::stod(splitString);
				getline(myFile, splitString);
				position_y = std::stod(splitString);

				measurement.raw_measurement << position_x, position_y;
			} else if(sensor_type == "RADAR") {
				measurement.sensor_type = Measurement::RADAR;
				measurement.raw_measurement = Eigen::VectorXd(3);

				getline(myFile, splitString,',');
				timestamp = std::stod(splitString);
				measurement.timestamp = timestamp;

				getline(myFile, splitString,',');
				radial_distance = std::stod(splitString);
				getline(myFile, splitString,',');
				angle = std::stod(splitString);
				std::getline(myFile, splitString);
				radial_velocity = std::stod(splitString);

				measurement.raw_measurement << radial_distance, angle, radial_velocity;
			}
			Eigen::VectorXd X_state(4);
			X_state = sensorFusion.ProcessMeasurement(measurement);
			myNewFile << measurement.timestamp << "," << X_state(0) << "," << X_state(1) << "," << X_state(2) << "," << X_state(3) << "," << std::endl;
		}
    }
    myFile.close();
	myNewFile.close();
}

KalmanFilter::KalmanFilter() {}
KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Initialize(Eigen::VectorXd &X_state_initial, Eigen::MatrixXd &P_Uncertainty_initial, Eigen::MatrixXd &F_state_transition_initial, 
							  Eigen::MatrixXd &H_measurement_function_initial, Eigen::MatrixXd &R_measurement_noise_initial, Eigen::MatrixXd &Q_process_noise_initial) {
	X_state = X_state_initial;
	P_Uncertainty = P_Uncertainty_initial;
	F_state_transition = F_state_transition_initial;
	H_measurement_function = H_measurement_function_initial;
	R_measurement_noise = R_measurement_noise_initial;
	Q_process_noise = Q_process_noise_initial;
}
Eigen::MatrixXd KalmanFilter::CalculateJacobian(const Eigen::VectorXd &X_state) {
	Eigen::MatrixXd H_measurement_function_jacobian(3, 4);

	auto position_x = X_state(0);
	auto position_y = X_state(1);
	auto velocity_x = X_state(2);
	auto velocity_y = X_state(3);

	auto position_xy_square = pow(position_x, 2) + pow(position_y, 2);
	auto position_xy_squareroot = hypot(position_x, position_y);

	H_measurement_function_jacobian << position_x/position_xy_squareroot, position_y/position_xy_squareroot, 0, 0,
									   -position_y/position_xy_square, position_x/position_xy_square, 0, 0,
									   position_y*(velocity_x*position_y - velocity_y*position_x)/pow(position_xy_squareroot, 3), position_x*(velocity_y*position_x - velocity_x*position_y)/pow(position_xy_squareroot, 3), position_x/position_xy_squareroot, position_y/position_xy_squareroot;
	return H_measurement_function_jacobian;
}
void KalmanFilter::Predict() {
	X_state = F_state_transition * X_state;
	Eigen::MatrixXd F_state_transition_transpose = F_state_transition.transpose();
	P_Uncertainty = F_state_transition * P_Uncertainty * F_state_transition_transpose + Q_process_noise; 
}
void KalmanFilter::Update(const Eigen::VectorXd &Z_measurement) {
	Eigen::VectorXd Z_measurement_prediction = H_measurement_function * X_state;
	Eigen::VectorXd Y_measurement_error = Z_measurement - Z_measurement_prediction;
	Eigen::MatrixXd H_measurement_function_transpose = H_measurement_function.transpose();
	Eigen::MatrixXd S_matrix = H_measurement_function * P_Uncertainty * H_measurement_function_transpose + R_measurement_noise;
	Eigen::MatrixXd S_matrix_inverse = S_matrix.inverse();
	Eigen::MatrixXd Kalman_gain = P_Uncertainty * H_measurement_function_transpose * S_matrix_inverse;
	X_state = X_state + (Kalman_gain * Y_measurement_error);
	double X_state_size = X_state.size();
	Eigen::MatrixXd Identity_matrix = Eigen::MatrixXd::Identity(X_state_size, X_state_size);
	P_Uncertainty = (Identity_matrix - Kalman_gain*H_measurement_function) * P_Uncertainty;
}
void KalmanFilter::UpdateEKF(const Eigen::VectorXd &Z_measurement) {
	double position_x = X_state(0);
	double position_y = X_state(1);
	double velocity_x = X_state(2);
	double velocity_y = X_state(3);

	double polar_cordinate1 = hypot(position_x, position_y);
	double polar_cordinate2 = atan2(position_y, position_x);
	double polar_cordinate3 = (position_x*velocity_x + position_y*velocity_y) / polar_cordinate1;
	Eigen::VectorXd Z_measurement_prediction(3);
	Z_measurement_prediction << polar_cordinate1, polar_cordinate2, polar_cordinate3;

	Eigen::VectorXd Y_measurement_error = Z_measurement - Z_measurement_prediction;
	Y_measurement_error(1) = atan2(sin(Y_measurement_error(1)), cos(Y_measurement_error(1)));

	Eigen::MatrixXd H_measurement_function_transpose = H_measurement_function.transpose();
	Eigen::MatrixXd S_matrix = H_measurement_function * P_Uncertainty * H_measurement_function_transpose + R_measurement_noise;
	Eigen::MatrixXd S_matrix_inverse = S_matrix.inverse();
	Eigen::MatrixXd Kalman_gain = P_Uncertainty * H_measurement_function_transpose * S_matrix_inverse;
	X_state = X_state + (Kalman_gain * Y_measurement_error);
	double X_state_size = X_state.size();
	Eigen::MatrixXd Identity_matrix = Eigen::MatrixXd::Identity(X_state_size, X_state_size);
	P_Uncertainty = (Identity_matrix - Kalman_gain*H_measurement_function) * P_Uncertainty;
}

SensorFusion::SensorFusion() {
	initialized = false;
	previous_timestamp = 0;

	R_measurement_noise_lidar = Eigen::MatrixXd(2, 2);
	R_measurement_noise_radar = Eigen::MatrixXd(3, 3);
	H_measurement_function_lidar = Eigen::MatrixXd(2, 4);

	R_measurement_noise_lidar << 0.0225, 0,
								 0, 0.0225;
	R_measurement_noise_radar << 0.09, 0, 0,
								 0, 0.009, 0,
								 0, 0, 0.09;
	H_measurement_function_lidar << 1, 0, 0, 0,
									0, 1, 0, 0;
}

SensorFusion::~SensorFusion() {}

Eigen::VectorXd SensorFusion::ProcessMeasurement(const Measurement &measurement) {
	if(!initialized) {
		EKF.X_state = Eigen::VectorXd(4);

		if(measurement.sensor_type == Measurement::LIDAR) {
			EKF.X_state << measurement.raw_measurement(0), measurement.raw_measurement(1), 0, 0;
		} else if(measurement.sensor_type == Measurement::RADAR) {
			double radial_distance = measurement.raw_measurement(0);
			double angle = measurement.raw_measurement(1);
			double radial_velocity = measurement.raw_measurement(2);

			double x_state = radial_distance * cos(angle);
			double y_state = radial_distance * sin(angle);
			double x_velocity = radial_velocity * cos(angle);
			double y_velocity = radial_velocity * sin(angle);

			EKF.X_state << x_state, y_state, x_velocity, y_velocity;
		}

		EKF.P_Uncertainty = Eigen::MatrixXd(4, 4);
    	EKF.P_Uncertainty << 1, 0, 0, 0,
               				 0, 1, 0, 0,
               				 0, 0, 1000, 0,
               				 0, 0, 0, 1000;

		initialized = true;
		previous_timestamp = measurement.timestamp;
		return EKF.X_state;
	}

	double dt = measurement.timestamp - previous_timestamp;
	previous_timestamp = measurement.timestamp;

	EKF.F_state_transition = Eigen::MatrixXd(4, 4);
	EKF.F_state_transition << 1, 0, dt, 0,
							  0, 1, 0, dt,
							  0, 0, 1, 0,
							  0, 0, 0, 1;
	
	double noise_ax = 9.0;
  	double noise_ay = 9.0;
  	double dt4 = (dt*dt*dt*dt)/4;
  	double dt3 = (dt*dt*dt)/2;
  	double dt2 = (dt*dt);

	EKF.Q_process_noise = Eigen::MatrixXd(4, 4);
	EKF.Q_process_noise << dt4*noise_ax, 0, dt3*noise_ax, 0,
            			   0, dt4*noise_ay, 0, dt3*noise_ay,
             			   dt3*noise_ax, 0, dt2*noise_ax, 0,
             			   0, dt3*noise_ay, 0, dt2*noise_ay;

	EKF.Predict();

	if (measurement.sensor_type == Measurement::RADAR) {
		EKF.H_measurement_function = EKF.CalculateJacobian(EKF.X_state);
		EKF.R_measurement_noise = R_measurement_noise_radar;
		EKF.UpdateEKF(measurement.raw_measurement);

	} else {
		EKF.H_measurement_function = H_measurement_function_lidar;
		EKF.R_measurement_noise = R_measurement_noise_lidar;
		EKF.Update(measurement.raw_measurement);
	}
	return EKF.X_state;
}

int main(int argc, char* argv[]) {
	CSV csv;
	csv.read_write_csv("../data/sensor_data.csv");
	return 0;
}

