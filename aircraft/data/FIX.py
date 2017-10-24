import os

for f in os.listdir("."):
    if f.endswith(".txt"):
	with open(f,"r") as file:	
            print("Working on file...")
            data = file.read()
            data = data.replace("Boeing 707","Boeing_707")
            data = data.replace("Boeing 717","Boeing_717")
            data = data.replace("Boeing 727","Boeing_727")
            data = data.replace("Boeing 737","Boeing_737")
            data = data.replace("Boeing 747","Boeing_747")
            data = data.replace("Boeing 757","Boeing_757")
            data = data.replace("Boeing 767","Boeing_767")
            data = data.replace("Boeing 777","Boeing_777")
            data = data.replace("BAE 146","BAE-146")
            data = data.replace("Beechcraft 1900","Beechcraft_1900")
            data = data.replace("Cessna 172","Cessna-172")
            data = data.replace("Cessna 208","Cessna-208")
            data = data.replace("Cessna Citation","Cessna_Citation")
            data = data.replace("Challenger 600","Challenger_600")
            data = data.replace("Dash 8","Dash_8")
            data = data.replace("Dornier 328","Dornier_328")
            data = data.replace("Embraer E-Jet","Embraer_E-Jet")
            data = data.replace("Embraer ERJ 145","Embraer_ERJ_145")
            data = data.replace("Embraer Legacy 600","Embraer_Legacy_600")
            data = data.replace("Eurofighter Typhoon","Eurofighter_Typhoon")
            data = data.replace("Falcon 2000","Falcon_2000")
            data = data.replace("Falcon 900","Falcon_900")
            data = data.replace("Fokker 100","Fokker_100")
            data = data.replace("Fokker 70","Fokker_70")
            data = data.replace("Fokker 50","Fokker_50")
            data = data.replace("Global Express","Global_Express")
            data = data.replace("Hawk T1","Hawk_T1")
            data = data.replace("King Air","King_Air")
            data = data.replace("Saab 2000","Saab_2000")
            data = data.replace("Saab 340","Saab_340")
        wr = open(f,"w")
        wr.write(data)

print("Operation Complete.\n")











	
