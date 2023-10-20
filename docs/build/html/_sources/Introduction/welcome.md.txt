## Motivation

### Background
The electrification of transportation is a key component of the decarbonization of the energy sector. 
The inevitable growth in EV demand is expected to increase the demand for electricity, which will require the 
expansion of the electricity grid, while leveraging new and existing technology in clever ways. Coordination and control
of distributed energy resources (DERs) will be key to the successful integration of electric vehicles (EVs) into the 
electricity grid. Today, there is a fundamental disconnect between the utilities and electric vehicle supply equipment
providers, which makes rapid deployment of EV charging infrastructure difficult. This work presents a co-simulation
platform that can serve as a testbed for design and optimization of EV charging stations, with the power grid in mind.

### Bridging the disconnect between utilities and EVSE providers

Say an EV charging operator (i.e., ChargePoint, Electrify America, etc.) wants to deploy a new charging station within
San Francisco, CA. The operator would need to contact the utility (i.e., PG&E) to determine the available capacity at 
the desired location. The utility would then need to perform a grid study to determine the available capacity at the 
desired location. This process can take months to complete sometimes. Eventually the utility might decide that a new 
power transformer is needed to support the new charging station (let's not even get into transformer supply-chain 
issues. However, with co-located DERs, such as solar PV and battery storage, the utility and EVSE operator
might be able to avoid upgrading the transformer. This is where *EV-EcoSim* comes in. Imagine if the EVSE operator can 
guarantee that the charging station will not exceed a certain power threshold, say 100 kW. Then,the utility can approve
the installment of the charging unit without the need for an upgrade. In fact, the utility and EVSE operator can come 
to an agreement where the EVSE operator can provide grid services to the utility, such as peak shaving, voltage regulation,
etc. This is the future of the electricity grid, and *EV-EcoSim* can help make this a reality.
