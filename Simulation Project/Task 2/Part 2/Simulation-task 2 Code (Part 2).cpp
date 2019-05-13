//This is Project 2 (Part-2) from Shashank Shekhar [200262327].
#include<iostream>
#include<fstream>
#include<tuple>
#include<bits/c++0x_warning.h>
#include<queue>
#include<stdlib.h>
#include<math.h>
#include<iomanip>
using namespace std;

void printval(ofstream &outfile, float mc, float cla, float cls, int buffer, priority_queue <double, vector<double>, greater<double> > clr){
	outfile<<mc<<";\t"<<cla<<";\t"<<cls<<";\t"<<buffer<<";\t";
	while(!clr.empty()){ 
        outfile<<clr.top()<<"  "; 
        clr.pop(); 
    } 
    outfile<<'\n'; 
}

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
	double mc, cla, inter_cla, cls, inter_cls, inter_clr, mc_end, rno;
	int buffer, max_buffer;
	char trace;
	priority_queue <double, vector<double>, greater<double> > clr;
		
	//Open the conection for outputting to a file. Overwrite the same file each time a write is made to the file.
	//Open file for output streaming
	ofstream outfile;
	outfile.open("output.txt", ios::trunc);
	//Set output file precision to 4
	outfile<<fixed;
    outfile<<setprecision(4);
    
    // Get all the required input
	if (argc != 6){
		//Give error message saying that argument count isn't correct
    	outfile<<"Check passed arguments count...";
		return -1;
	}
	else{
		// Take input from terminal
		inter_cla = atof(argv[1]); inter_clr = atof(argv[2]); inter_cls = atof(argv[3]); max_buffer = atoi(argv[4]); mc_end = atof(argv[5]);
	}
	
	//Now fix the seed value for generating psuedo-random numbers
	// Use 5 as fixed seed and then generate random numbers in range [0,1]
    srand(5); 
    rno = double(rand()) / (double(RAND_MAX));
	//Now start with the simulation process
	//Assign Machine clock to 0 and cls = -1 (indicating that the service time is unknown) and buffer = 0. Also, calculate cla
	mc = 0; cls = -1; buffer = 0; cla = (-1)*inter_cla*log(rno);
	//Now start the simulation process
	while(mc<=mc_end){
		printval(outfile, mc, cla, cls, buffer, clr);
		//Get minimum of cla, cls and front of clr. Assign the value to minval
		pair<double, char> p = findmin(cla, cls, clr);
		//Change mc to the minval obtained from pair and update either cla, cls, clr and/or buffer depending on the second value of pair
		mc = p.first; trace = p.second;
		if(trace == 'a' && !buffer){ //meaning there is nothing to process and hence the cls is unknown
			//max_buffer will be minimum 1 and buffer will always start from 0. Therefore, if buffer is 0, it will always be less than max_buffer
			//Meaning, there is no need to check for overflow here
			rno = double(rand()) / (double(RAND_MAX));
			cla+=((-1)*inter_cla*log(rno)); cls = mc+inter_cls; buffer+=1;
		}
		else if(trace == 'a'){
			//Check for buffer overflow here
			if(buffer<max_buffer){
				rno = double(rand()) / (double(RAND_MAX));
				cla+=((-1)*inter_cla*log(rno)); buffer+=1; 
			}
			else{
				rno = double(rand()) / (double(RAND_MAX));
				cla+=((-1)*inter_cla*log(rno)); 
				rno = double(rand()) / (double(RAND_MAX));
				clr.push(mc+((-1)*inter_clr*log(rno)));
			}
		}
		else if(trace == 's'){
			cls+=inter_cls; buffer-=1;
			//Check for buffer underflow here
			if(!buffer){
				cls = -1;
			}
		}
		else if(trace == 'r' && !buffer){
			//Confirm this condition once if it can ever happen or not
			cls = mc+inter_cls; buffer+=1;
		}
		else if(trace == 'r'){
			//Check for buffer overflow here
			if(buffer<max_buffer){
				buffer+=1;
			}
			else{
				rno = double(rand()) / (double(RAND_MAX));
				clr.push(mc+((-1)*inter_clr*log(rno)));
			}
		}
	}
	//Close the file stream to prevent any accidental writing to the file
	outfile.close();
	return 0;
}
