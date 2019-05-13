//This is Project 1 (Task - 3) from Shashank Shekhar [200262327].
#include <bits/stdc++.h>
#include<iostream>
//#include<fstream>
#include<queue>
#include<tuple>
#include<stdlib.h>
#include<math.h>
#include<iomanip>
#include<random>
#include<functional>
#include<ctime>
#include<limits>
#define MAXSIZE 50001
#define BATCHSIZE 1000
#define BATCHES 50
using namespace std;

struct ret_packets_comparator
{
    bool operator()(pair<double, double> p1, pair<double, double> p2) {
        return p1.second <= p2.second;
    }
};

pair<double, char> findmin(double cla, double cls, priority_queue <double, vector<double>, greater<double> > &clr){
	double minval, curr_clr; char trace;
	// minval holds the min value which is returned by this function. trace keeps the track of the fact the minval came from where
	//If minval came from cla, then trace contains 'a', if minval came from cls, then trace contains 's', if minval came from curr_clr, then trace contains 'r'
	if(clr.empty()){
		if(cls != -1){
			if(cla <= cls){
				minval = cla;
				trace = 'a';
			}
			else{
				minval = cls;
				trace = 's';
			}
		}
		else{
			minval = cla;
			trace = 'a';
		}
	}
	else{
		//Assign the front value of the queue as the current value and then pop that value from the queue
		curr_clr = clr.top();
		if(cls != -1){
			if(curr_clr<=cla && curr_clr<=cls){
				minval = curr_clr;
				trace = 'r';
				//Pop the front element from the queue and return
				clr.pop();
			}
			else if(cla<=cls){
				minval = cla;
				trace = 'a';
			}
			else{
				minval = cls;
				trace = 's';
			}
		}
		else{
			if(curr_clr<=cla){
				minval = curr_clr;
				trace = 'r';
				//Pop the front element from the queue and return
				clr.pop();
			}
			else{
				minval = cla;
				trace = 'a';
			}
		}
	}
	return make_pair(minval,trace);
}

int main(int argc, char* argv[]){
	double mc, cla, mean_inter_cla, cls, inter_cls, mean_inter_clr, t_sum_of_means = 0, d_sum_of_means = 0, t_mean_of_means, d_mean_of_means;
	double t_mean_standard_dev, d_mean_standard_dev, t_mean_ci_min, t_mean_ci_max, d_mean_ci_min, d_mean_ci_max;
	double t_95_standard_dev, d_95_standard_dev, t_95_ci_min, t_95_ci_max, d_95_ci_min, d_95_ci_max, t_sum_of_95s = 0, d_sum_of_95s = 0;
	double t_mean_of_95s, d_mean_of_95s;
	long long int buffer, max_buffer, y; //y is used throughout the program as a temp variable for iteration etc.
	char trace;
	//Keeping 2 array of pairs
	//t_pairs and d_pairs hold the time to certify (T) and the orbiting time (D) for first 50000 entries
	vector< tuple<double,double,double> > t_pairs;
	vector< tuple<double,double,double> > d_pairs;
	//Vectors containing t_means and d_means
	vector<double> t_means, d_means;
	//Vectors containing t_95 percentile and d_95 percentile
	vector<double> t_95s, d_95s;
	//Temporary vectors for intermediate operations
	vector<double> t_temp_vect, d_temp_vect;
	//These represent counts of t_pairs and d_pairs
	long long int t_count = 0, d_count = 0; 
	//This the priority queue representing the arrival clock time of the packets that are orbiting / got retransmitted
	priority_queue <double, vector<double>, greater<double> > clr;
	//Below is a simple queue to keep track of the original packets in the buffer. Using queue since they will always be processed in order of 
	//their arrival with no preference to any packet
	queue< pair<double, double> > buf_packets;
	//Below is the priority queue to keep track of the orbiting / retransmitted packet requests. Using priority queue since the newly generated
	//orbiting time may be smaller than the previous one
	priority_queue < pair<double, double>, vector< pair<double, double> >, ret_packets_comparator > ret_packets;
	pair<double, double> buf_prev_element; // This keeps track of the last element in the buf_packets queue. When a new element is pushed, this
	//value is updated.
		
	//Open the conection for outputting to a file. Overwrite the same file each time a write is made to the file.
	//Open file for output streaming
//	ofstream outfile;
//	outfile.open("output.txt", ios::app);
	//Set output file precision to 4
//	outfile<<fixed;
//    outfile<<setprecision(4);
    
    // Get all the required input
	if (argc != 5){
		//Give error message saying that argument count isn't correct
    	cout<<"Check passed arguments count...";
		return -1;
	}
	else{
		// Take input from terminal
		mean_inter_cla = atof(argv[1]); mean_inter_clr = atof(argv[2]); inter_cls = atof(argv[3]); max_buffer = atoi(argv[4]);
	}
	
	//Pseudo random number generator using fixed seed value of 2
    auto rno = bind(uniform_real_distribution<double>(numeric_limits< double >::min(), 1), mt19937(2));	double temp_no;
    
	//Now start with the simulation process
	//Assign Machine clock to 0 and cls = -1 (indicating that the service time for the 1st arrival is unknown) and buffer = 0 (No elements in the buffer). 
	mc = 0; cls = -1; buffer = 0;
	//Now, Calculate cla (arrival for the first request)
	cla = (-1)*mean_inter_cla*log(rno());
	//Initial setup of the simulation is done. Now, start with the simulation process
	clock_t begin = clock();
	while(t_count <= 50000 || d_count <= 50000){
		//Get minimum of cla, cls and front of clr. Assign the value to minval
		pair<double, char> p = findmin(cla, cls, clr);
		
		//Change mc to the minval obtained from pair and update either cla, cls, clr and/or buffer depending on the second value of pair
		mc = p.first; trace = p.second;
		//If there is nothing to process at present
		if(trace == 'a' && !buffer){ //meaning there is nothing to process and hence the cls is unknown
			//max_buffer will be minimum 1 and buffer will always start from 0. Therefore, if buffer is 0, it will always be less than max_buffer
			//Meaning, there is no need to check for overflow here
			cla+=((-1)*mean_inter_cla*log(rno())); cls = mc+inter_cls; buffer+=1;
			//Now update the buf_packets queue
			buf_packets.push(make_pair(mc, mc+inter_cls));
			buf_prev_element = make_pair(mc, mc+inter_cls);
			//Since there was no retransmission, directly add 0 orbit time for this packet into the d_pairs tuple
			if(d_count <= MAXSIZE){
				d_pairs.push_back(make_tuple(mc, mc, 0));
				d_count++;
			}
		}//If there is atleast one packet in processing state
		else if(trace == 'a'){
			//Check for buffer overflow here
			if(buffer<max_buffer){
				cla+=((-1)*mean_inter_cla*log(rno())); buffer+=1; 
				//Now update the buf_packets queue using the value of the buf_prev_element
				buf_packets.push(make_pair(mc, buf_prev_element.second+inter_cls));
				buf_prev_element = make_pair(mc, buf_prev_element.second+inter_cls);
				//Since there was no retransmission, directly add 0 orbit time for this packet into the d_pairs tuple
				if(d_count <= MAXSIZE){
					d_pairs.push_back(make_tuple(mc, mc, 0));
					d_count++;
				}
			}
			else{
				cla+=((-1)*mean_inter_cla*log(rno())); 
				temp_no = rno();
				clr.push(mc+((-1)*mean_inter_clr*log(temp_no)));
				//Now update the ret_packets queue
				ret_packets.push(make_pair(mc, mc+((-1)*mean_inter_clr*log(temp_no))));
			}
		}
		else if(trace == 's'){
			cls+=inter_cls; buffer-=1;
			//Check for buffer underflow here
			if(!buffer){
//				cout<<"I entered here...\t";
				cls = -1;
			}
			//Now, first assign the front element of buf_packets to t_pairs and then pop the front element from the buf_packets queue
			if(t_count <= MAXSIZE){
				t_pairs.push_back(make_tuple(buf_packets.front().first, buf_packets.front().second, buf_packets.front().second - buf_packets.front().first)); 
				t_count++;
			}
			buf_packets.pop();
		}//Orbiting packet returned and directly went to processor for service
		else if(trace == 'r' && !buffer){
			//Confirm this condition once if it can ever happen or not
			cls = mc+inter_cls; buffer+=1;
			if(d_count <= MAXSIZE){
				d_pairs.push_back(make_tuple(ret_packets.top().first, ret_packets.top().second, ret_packets.top().second - ret_packets.top().first));
				d_count++;
			}
			//Before poping the element from the ret_packets, insert it into the buf_packets queue and update the buf_prev_element and then pop it
			buf_packets.push(make_pair(ret_packets.top().first, mc+inter_cls));
			buf_prev_element = make_pair(ret_packets.top().first, mc+inter_cls);
			//Now pop the element from the ret_packets queue
			ret_packets.pop();
		}//Orbiting packet returned, maybe got added to queue or had to re-orbit
		else if(trace == 'r'){
			//Check for buffer overflow here (Got added to queue)
			if(buffer<max_buffer){
				buffer+=1;
				if(d_count <= MAXSIZE){
					d_pairs.push_back(make_tuple(ret_packets.top().first, ret_packets.top().second, ret_packets.top().second - ret_packets.top().first));
					d_count++;
				}
				//Before poping the element from the ret_packets, insert it into the buf_packets queue and update the buf_prev_element and then pop it
				buf_packets.push(make_pair(ret_packets.top().first, buf_prev_element.second+inter_cls));
				buf_prev_element = make_pair(ret_packets.top().first, buf_prev_element.second+inter_cls);
				ret_packets.pop();
			}// Had to re-orbit
			else{
				temp_no = rno();
				clr.push(mc+((-1)*mean_inter_clr*log(temp_no)));
				//Update the ret_packets queue. But since this is the re-entry of the packet which was previously orbiting, we need to update 
				//the old value rather than creating a new entry. Since priority queue doesn't allow a direct update, remove and re-insert
				double temp = ret_packets.top().first;
				ret_packets.pop();
				ret_packets.push(make_pair(temp, mc+((-1)*mean_inter_clr*log(temp_no))));
			}
		}
	}
	//Once all the 50 K entries have been generated, then find mean for both T and D
	//----------------------------------------------------------------------------------------
	for(long long int i = 1; i<BATCHES; i++){
		double t_sum = 0, d_sum = 0;
		for(long long int j = i*BATCHSIZE; j<((i*BATCHSIZE) + BATCHSIZE); j++){
			t_sum+=(get<2>(t_pairs[j]));
			d_sum+=(get<2>(d_pairs[j]));
		}
		t_means.push_back(t_sum/BATCHSIZE);
		d_means.push_back(d_sum/BATCHSIZE);
		t_sum_of_means+=(t_sum/BATCHSIZE);
		d_sum_of_means+=(d_sum/BATCHSIZE);
	}
	t_mean_of_means = t_sum_of_means / (BATCHES-1);
	d_mean_of_means = d_sum_of_means / (BATCHES-1);
	
	double t_temp = 0, d_temp = 0;
	//Getting Summation (x(i)-X(bar)) now for both T and D
	for(vector<double>::iterator it = t_means.begin(); it != t_means.end(); it++){
		t_temp+=((*it-t_mean_of_means)*(*it-t_mean_of_means));
	}
	t_mean_standard_dev = ((double)1/(BATCHES-2))*t_temp;
	t_mean_ci_min = t_mean_of_means-(1.68*(t_mean_standard_dev/sqrt(BATCHES-1)));
	t_mean_ci_max = t_mean_of_means+(1.68*(t_mean_standard_dev/sqrt(BATCHES-1)));
	for(vector<double>::iterator it = d_means.begin(); it != d_means.end(); it++){
		d_temp+=((*it-d_mean_of_means)*(*it-d_mean_of_means));
	}
	d_mean_standard_dev = ((double)1/(BATCHES-2))*d_temp;
	d_mean_ci_min = d_mean_of_means-(1.68*(d_mean_standard_dev/sqrt(BATCHES-1)));
	d_mean_ci_max = d_mean_of_means+(1.68*(d_mean_standard_dev/sqrt(BATCHES-1)));	
	
	cout<<"Given:\n";
	cout<<"1) mean inter-arrival time = "<<mean_inter_cla<<"\n";
	cout<<"2) mean orbiting time = "<<mean_inter_clr<<"\n";
	cout<<"3) service time = "<<inter_cls<<"\n";
	cout<<"4) buffer size = "<<max_buffer<<"\n";
	
	cout<<"CI range for T (Mean) = "<<t_mean_ci_min<<" to "<<t_mean_ci_max<<"\n";
	cout<<"CI range for D (Mean) = "<<d_mean_ci_min<<" to "<<d_mean_ci_max<<"\n";

//	outfile<<"t_mean_of_means: "<<t_mean_of_means<<"\n";
//	outfile<<"d_mean_of_means: "<<d_mean_of_means<<"\n";
	//-------------------------------------------------------------------------------------------
	
	//Now, working on the 95th percentile
	for(long long int i = 1; i<BATCHES; i++){
		t_temp_vect.clear(); d_temp_vect.clear();
		for(long long int j = i*BATCHSIZE; j<((i*BATCHSIZE) + BATCHSIZE); j++){
			t_temp_vect.push_back(get<2>(t_pairs[j]));
			d_temp_vect.push_back(get<2>(d_pairs[j]));
		}
		sort(t_temp_vect.begin(), t_temp_vect.end());
		sort(d_temp_vect.begin(), d_temp_vect.end());
		t_95s.push_back(t_temp_vect[ceil(0.95*BATCHSIZE)]);
		d_95s.push_back(d_temp_vect[ceil(0.95*BATCHSIZE)]);
		t_sum_of_95s+=(t_temp_vect[ceil(0.95*BATCHSIZE)]);
		d_sum_of_95s+=(d_temp_vect[ceil(0.95*BATCHSIZE)]);
	}
	t_mean_of_95s = t_sum_of_95s / (BATCHES - 1);
	d_mean_of_95s = d_sum_of_95s / (BATCHES - 1);
	
	t_temp = 0, d_temp = 0;
	for(vector<double>::iterator it = t_95s.begin(); it != t_95s.end(); it++){
		t_temp+=((*it-t_mean_of_95s)*(*it-t_mean_of_95s));
	}
	t_95_standard_dev = ((double)1/(BATCHES-2))*t_temp;
	t_95_ci_min = t_mean_of_95s-(1.68*(t_95_standard_dev/sqrt(BATCHES-1)));
	t_95_ci_max = t_mean_of_95s+(1.68*(t_95_standard_dev/sqrt(BATCHES-1)));
	for(vector<double>::iterator it = d_95s.begin(); it != d_95s.end(); it++){
		d_temp+=((*it-d_mean_of_95s)*(*it-d_mean_of_95s));
	}
	d_95_standard_dev = ((double)1/(BATCHES-2))*d_temp;
	d_95_ci_min = d_mean_of_95s-(1.68*(d_95_standard_dev/sqrt(BATCHES-1)));
	d_95_ci_max = d_mean_of_95s+(1.68*(d_95_standard_dev/sqrt(BATCHES-1))); 
	
	cout<<"CI range for T (95th percentile) = "<<t_95_ci_min<<" to "<<t_95_ci_max<<"\n";
	cout<<"CI range for D (95th percentile) = "<<d_95_ci_min<<" to "<<d_95_ci_max<<"\n";

//	outfile<<"t_mean_of_95s: "<<t_mean_of_95s<<"\n";
//	outfile<<"d_mean_of_95s: "<<d_mean_of_95s<<"\n";
	//-------------------------------------------------------------------------------------------
//	outfile<<"\n\n\n";
	
	//Close the file stream to prevent any accidental writing to the file
//	outfile.close();
	return 0;
}
