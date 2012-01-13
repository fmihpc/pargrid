#ifndef PARGRID_H
#define PARGRID_H

// Includes for standard headers:

#include <cstdlib>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <set>
#include <stdint.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <sstream>

// Includes for external package headers:

#include <mpi.h>
#include <zoltan_cpp.h>

#include "mpiconversion.h"

#ifdef PROFILE
   #include "profiler.h"
#endif

namespace pargrid {
   typedef float CellWeight;                        /**< Datatype Zoltan uses for cell and edge weights.*/
   typedef double CellCoordinate;                   /**< Datatype Zoltan uses for cell coordinates.*/
   typedef unsigned int CellID;                     /**< Datatype used for global cell IDs.*/
   typedef int MPI_processID;                       /**< Datatype MPI uses for process IDs.*/
   typedef unsigned char nbrIDtype;

   enum InputParameter {
      cellWeightScale,                              /**< Value used for cell weights.*/
      edgeWeightScale,                              /**< Value (or its scale) used to calculate edge weights.*/
      imbalanceTolerance,                           /**< Load imbalance tolerance value.*/
      loadBalancingMethod,                          /**< Load balancing method to use.*/
      processesPerPartition                         /**< Processes per partition (hierarchical partitioning only).*/
   };

   enum StencilType {
      localToRemoteUpdates,
      remoteToLocalUpdates
   };
   
   enum PartitioningMode {
      partition,
      repartition,
      refine
   };
   
   template<class C> class ParGrid;
   
   // ************************************* //
   // ***** ZOLTAN CALLBACK FUNCTIONS ***** //
   // ************************************* //

   template<class C>
   void cb_getAllCellCoordinates(void* pargrid,int N_globalEntries,int N_localEntries,int N_cellIDs,ZOLTAN_ID_PTR globalID,
				 ZOLTAN_ID_PTR localID,int N_coords,double* geometryData,int* rcode);   
   template<class C>
   void cb_getCellCoordinates(void* pargrid,int N_globalEntries,int N_localEntries,ZOLTAN_ID_PTR globalID,
			   ZOLTAN_ID_PTR localID,double* geometryData,int* rcode);
   template<class C>
   void cb_getAllCellEdges(void* pargrid,int N_globalIDs,int N_localIDs,int N_cells,ZOLTAN_ID_PTR globalIDs,
			   ZOLTAN_ID_PTR localIDs,int* N_edges,ZOLTAN_ID_PTR nbrGlobalIDs,int* nbrHosts,
			   int N_weights,CellWeight* edgeWeights,int* rcode);
   template<class C>
   void cb_getCellEdges(void* pargrid,int N_globalIDs,int N_localIDs,ZOLTAN_ID_PTR globalID,
			ZOLTAN_ID_PTR localID,ZOLTAN_ID_PTR nbrGlobalIDs,int* nbrHosts,
			int N_weights,CellWeight* weight,int* rcode);
   template<class C>
   void cb_getHierarchicalParameters(void* pargrid,int level,Zoltan_Struct* zs,int* rcode);
   template<class C>
   int cb_getHierarchicalPartNumber(void* pargrid,int level,int* rcode);
   template<class C>
   void cb_getHyperedges(void* pargrid,int N_globalIDs,int N_vtxedges,int N_pins,int format,
			 ZOLTAN_ID_PTR vtxedge_GID,int* vtxedge_ptr,ZOLTAN_ID_PTR pin_GID,int* rcode);
   template<class C>
   void cb_getHyperedgeWeights(void* pargrid,int N_globalIDs,int N_localIDs,int N_edges,int N_weights,
			       ZOLTAN_ID_PTR edgeGlobalID,ZOLTAN_ID_PTR edgeLocalID,CellWeight* edgeWeights,int* rcode);
   template<class C>
   int cb_getMeshDimension(void* pargrid,int* rcode);
   template<class C>
   void cb_getLocalCellList(void* pargrid,int N_globalIDs,int N_localIDs,ZOLTAN_ID_PTR globalIDs,
			    ZOLTAN_ID_PTR localIDs,int N_weights,CellWeight* cellWeights,int* rcode);
   template<class C>
   void cb_getNumberOfAllEdges(void* pargrid,int N_globalIDs,int N_localIDs,int N_cells,ZOLTAN_ID_PTR globalIDs,
			      ZOLTAN_ID_PTR localIDs,int* N_edges,int* rcode);
   template<class C>
   int cb_getNumberOfEdges(void* pargrid,int N_globalIDs,int N_localIDs,ZOLTAN_ID_PTR globalID,
			ZOLTAN_ID_PTR localID,int* rcode);
   template<class C>
   int cb_getNumberOfHierarchicalLevels(void* pargrid,int* rcode);
   template<class C>
   void cb_getNumberOfHyperedges(void* pargrid,int* N_lists,int* N_pins,int* format,int* rcode);
   template<class C>
   void cb_getNumberOfHyperedgeWeights(void* pargrid,int* N_edges,int* rcode);
   template<class C>
   int cb_getNumberOfLocalCells(void* pargrid,int* rcode);

   // ******************************
   // ***** CLASS DECLARATIONS *****
   // ******************************

   // ***** PARCELL *****
   
   template<class C> struct ParCell {
    public:
      ParCell();
      ~ParCell();

      unsigned char info[2];                       /**< N_neighbours, refinementLevel.*/
      std::vector<CellID> neighbours;              /**< Array of size N_neighbours, neighbour local IDs.*/
      std::vector<unsigned char> neighbourTypes;   /**< Array of size N_neighbours, neighbour type IDs.*/
      std::vector<unsigned char> refNumbers;       /**< Array of size refinementLevel, cell refinement numbers.*/
      uint32_t neighbourFlags;                     /**< Neighbour existence flags. Each cell existing within a 3x3x3 
						    * cube of cells, in which this cells sits at the center, has its 
						    * corresponding bit in neighbourFlags flipped to unit value.
						    * This variable has undefined values for remote cells.
						    */
      C userData;                                  /**< User data.*/

      void getData(bool sending,int ID,int receivesPosted,int receiveCount,int* blockLengths,MPI_Aint* displacements,MPI_Datatype* types);
      unsigned int getDataElements(int ID) const;
      void getMetadata(int ID,int* blockLengths,MPI_Aint* displacements,MPI_Datatype* types);
      unsigned int getMetadataElements(int ID) const;
      
    private:

   };

   // ***** STENCIL *****
   
   template<class C>
   struct Stencil {
    public:
      Stencil();
      ~Stencil();
      
      bool addTransfer(int ID,bool recalculate);
      void clear();
      const std::vector<CellID>& getBoundaryCells() const;
      const std::vector<CellID>& getInnerCells() const;
      bool initialize(ParGrid<C>& pargrid,StencilType stencilType,const std::vector<unsigned char>& receives);
      bool removeTransfer(int ID);
      bool startTransfer(int ID);
      bool update();
      bool wait(int ID);

    private:
      bool calcLocalUpdateSendsAndReceives();
      bool calcTypeCache(int ID);

      /** Wrapper for MPI_Datatype. This was created to make copying and deletion of
       * MPI_Datatype work correctly. Calls MPI_Type_dup in appropriate places.*/
      struct TypeWrapper {
	 MPI_Datatype type;                               /**< MPI datatype.*/
	 
	 TypeWrapper();
	 TypeWrapper(const TypeWrapper& tw);
	 ~TypeWrapper();
	 TypeWrapper& operator=(const TypeWrapper& tw);
      };
      
      /** MPI Datatype cache for data transfer.*/
      struct TypeCache {
	 std::vector<TypeWrapper> recvs; /**< MPI datatypes for receiving data.*/
	 std::vector<TypeWrapper> sends; /**< MPI datatypes for sending data.*/
      };

      /** Information on data transfer.*/
      struct TypeInfo {
	 int N_receives;         /**< Total number of messages received during this transfer.*/
	 int N_sends;            /**< Total number of messages sent during this transfer.*/
	 bool typeVolatile;      /**< If true, MPI Datatypes need to be recalculated every time 
				  * this transfer is started.*/
	 bool started;           /**< If true, this transfer has started and MPI requests are valid.*/
	 MPI_Request* requests;  /**< MPI requests associated with this transfer.*/
      };
      
      std::vector<CellID> boundaryCells;                           /**< List of boundary cells of this stencil.*/
      std::vector<CellID> innerCells;                              /**< List of inner cells of this stencil.*/
      bool initialized;                                            /**< If true, Stencil has initialized successfully and is ready for use.*/
      std::vector<unsigned char> receivedNbrTypeIDs;               /**< Neighbour type IDs indicating which cells to receive data from.*/
      std::vector<unsigned char> sentNbrTypeIDs;                   /**< Neighbour type IDs indicating which cells to send data.*/
      std::map<int,std::map<MPI_processID,TypeCache> > typeCaches; /**< MPI datatype caches for each transfer identifier,
								    * one cache per neighbouring process.*/
      std::map<int,TypeInfo> typeInfo;                             /**< Additional data transfer information for each 
								    * transfer identifier.*/
      ParGrid<C>* parGrid;                                         /**< Pointer to parallel grid.*/
      StencilType stencilType;
      std::map<MPI_processID,std::set<CellID> > recvs;             /**< List of received cells (ordered by global ID) from each neighbour process.*/
      std::map<MPI_processID,std::set<CellID> > sends;             /**< List of cells sent (ordered by global ID) to each neighbour process.*/
      std::map<CellID,std::set<MPI_processID> > recvCounts;        /**< For each local cell, identified by global ID, ranks of remote 
								    * processes whom to receive an update from. This map is only 
								    * used for stencilType remoteToLocalUpdates.*/
   };

   // ***** PARGRID *****

   /**
    * User-defined template class must contain following member functions:
    * void getCoordinates(CellCoordinate*) const;
    * CellWeight getWeight() const;
    */
   template<class C> class ParGrid {
    public:
      ParGrid();
      ~ParGrid();

      bool addCell(CellID cellID,const std::vector<CellID>& nbrIDs,const std::vector<nbrIDtype>& nbrTypes);
      bool addCellFinished();
      int addStencil(pargrid::StencilType stencilType,const std::vector<unsigned char>& recvNbrTypeIDs);
      bool addTransfer(unsigned int stencil,int identifier,bool recalculate);
      bool balanceLoad();
      void barrier(int threadID=0) const;
      void calcNeighbourOffsets(unsigned char nbrTypeID,int& i_off,int& j_off,int& k_off) const;
      unsigned char calcNeighbourTypeID(int i_off,int j_off,int k_off) const;
      bool checkPartitioningStatus(int& counter) const;
      bool finalize();
      const std::vector<CellID>& getBoundaryCells(unsigned int stencil,int identifier) const;
      std::vector<CellID>& getCellNeighbourIDs(CellID cellID);
      MPI_Comm getComm() const;
      std::vector<CellID>& getExteriorCells();
      const std::vector<CellID>& getGlobalIDs() const;
      const std::vector<MPI_processID>& getHosts() const;
      bool getInitialized() const;
      const std::vector<CellID>& getInnerCells(unsigned int stencil,int identifier) const;
      std::vector<CellID>& getInteriorCells();
      CellID getLocalID(CellID globalID) const;
      uint32_t getNeighbourFlags(CellID cellID) const;
      const std::set<MPI_processID>& getNeighbourProcesses() const;
      CellID getNumberOfAllCells() const;
      CellID getNumberOfLocalCells() const;
      MPI_processID getProcesses() const;
      MPI_processID getRank() const;
      bool getRemoteNeighbours(CellID cellID,const std::vector<unsigned char>& nbrTypeIDs,std::vector<CellID>& nbrIDs);
      bool initialize(MPI_Comm comm,const std::vector<std::map<InputParameter,std::string> >& parameters,int threadID=0,int mpiThreadingLevel=0);
      bool initialLoadBalance(bool balanceLoad=true);
      CellID invalid() const;
      bool localCellExists(CellID cellID);
      C* operator[](const CellID& cellID);
      bool setPartitioningMode(PartitioningMode pm);
      bool startNeighbourExchange(unsigned int stencil,int identifier);
      bool wait(unsigned int stencil,int identifier);

      // ***** ZOLTAN CALLBACK FUNCTION DECLARATIONS *****

      void getAllCellCoordinates(int N_globalEntries,int N_localEntries,int N_cellIDs,ZOLTAN_ID_PTR globalID,
				 ZOLTAN_ID_PTR localID,int N_coords,double* geometryData,int* rcode);
      void getCellCoordinates(int N_globalEntries,int N_localEntries,ZOLTAN_ID_PTR globalID,
			      ZOLTAN_ID_PTR localID,double* geometryData,int* rcode);
      void getAllCellEdges(int N_globalIDs,int N_localIDs,int N_cells,ZOLTAN_ID_PTR globalIDs,
			   ZOLTAN_ID_PTR localIDs,int* N_edges,ZOLTAN_ID_PTR nbrGlobalIDs,int* nbrHosts,
			   int N_weights,CellWeight* edgeWeights,int* rcode);
      void getCellEdges(int N_globalIDs,int N_localIDs,ZOLTAN_ID_PTR globalID,
			ZOLTAN_ID_PTR localID,ZOLTAN_ID_PTR nbrGlobalIDs,int* nbrHosts,
			int N_weights,CellWeight* weight,int* rcode);
      void getHierarchicalParameters(int level,Zoltan_Struct* zs,int* rcode);
      int getHierarchicalPartNumber(int level,int* rcode);
      void getHyperedges(int N_globalIDs,int N_vtxedges,int N_pins,int format,
			 ZOLTAN_ID_PTR vtxedge_GID,int* vtxedge_ptr,ZOLTAN_ID_PTR pin_GID,int* rcode);
      void getHyperedgeWeights(int N_globalIDs,int N_localIDs,int N_edges,int N_weights,
			       ZOLTAN_ID_PTR edgeGlobalID,ZOLTAN_ID_PTR edgeLocalID,CellWeight* edgeWeights,int* rcode);
      void getLocalCellList(int N_globalIDs,int N_localIDs,ZOLTAN_ID_PTR globalIDs,
			    ZOLTAN_ID_PTR localIDs,int N_weights,CellWeight* cellWeights,int* rcode);
      int getMeshDimension(int* rcode);
      void getNumberOfAllEdges(int N_globalIDs,int N_localIDs,int N_cells,ZOLTAN_ID_PTR globalIDs,
			      ZOLTAN_ID_PTR localIDs,int* N_edges,int* rcode);
      int getNumberOfEdges(int N_globalIDs,int N_localIDs,ZOLTAN_ID_PTR globalID,
			   ZOLTAN_ID_PTR localID,int* rcode);
      int getNumberOfHierarchicalLevels(int* rcode);
      void getNumberOfHyperedges(int* N_lists,int* N_pins,int* format,int* rcode);
      void getNumberOfHyperedgeWeights(int* N_edges,int* rcode);
      int getNumberOfLocalCells(int* rcode);

    private:
      ParGrid(const ParGrid<C>& p);

      bool recalculateExteriorCells;                                      /**< If true, exteriorCells needs to be recalculated when 
									   * it is requested.*/
      bool recalculateInteriorCells;                                      /**< If true, interiorCells needs to be recalculated 
									   * when it is requested.*/
      std::vector<CellID> exteriorCells;                                  /**< List of exterior cells on this process, i.e. cells that lie on 
									   * the boundary of the simulation domain. These cells have one or more 
									   * missing neighbours. This list is recalculated only when needed.*/
      std::vector<CellID> interiorCells;                                  /**< List of interior cells on this process, i.e. cells with 
									   * zero missing neighbours. This list is recalculated only when needed.*/      
      CellWeight cellWeight;                                              /**< Cell weight scale, used to calculate cell weights for Zoltan.*/
      bool cellWeightsUsed;                                               /**< If true, cell weights are calculated.*/
      
      std::vector<ParCell<C> > cells;                                     /**< Local cells stored on this process, followed by remote cells.*/
      std::vector<CellID> globalIDs;                                      /**< Global IDs of cells stored in vector cells.*/
      CellID N_localCells;                                                /**< Number of local cells on this process.*/
      std::map<CellID,CellID> global2LocalMap;                            /**< Global-to-local ID mapping.*/
      std::vector<MPI_processID> hosts;                                   /**< Host process for each cell in vector cells.*/
      
      MPI_Comm comm;                                                      /**< MPI communicator used by ParGrid.*/
      CellWeight edgeWeight;                                              /**< Edge weight scale, used to calculate edge weights for Zoltan.*/
      bool edgeWeightsUsed;                                               /**< If true, edge weights are calculated.*/
      bool initialized;                                                   /**< If true, ParGrid initialized correctly and is ready for use.*/
      std::vector<std::list<std::pair<std::string,std::string> > > 
	                                         loadBalancingParameters; /**< LB parameters for Zoltan for each hierarchical level.*/
      int mpiThreadingLevel;                                              /**< Threading level in which MPI was initialized.*/
      MPI_processID myrank;                                               /**< MPI rank of this process in communicator comm.*/
      std::set<MPI_processID> nbrProcesses;                               /**< MPI ranks of neighbour processes. A process is considered 
									   * to be a neighbour if it has any of this processes local 
									   * cells' neighbours. Calculated in balanceLoad.*/
      MPI_processID N_processes;                                          /**< Number of MPI processes in communicator comm.*/
      int partitioningCounter;
      std::vector<MPI_Request> recvRequests;
      std::vector<MPI_Request> sendRequests;
      std::map<unsigned int,Stencil<C> > stencils;
      Zoltan* zoltan;                                                     /**< Pointer to Zoltan.*/

      #ifdef PROFILE
         int profZoltanLB;
         int profParGridLB;
         int profMPI;
         int profTotalLB;
      #endif
      
      bool checkInternalStructures() const;
      void invalidate();
      bool syncCellHosts();
   };

   // *************************************************
   // ***** PARCELL TEMPLATE FUNCTION DEFINITIONS *****
   // *************************************************
   
   template<class C>
   ParCell<C>::ParCell(): neighbourFlags(0) {
      info[0] = 0;
      info[1] = 1;
      refNumbers.resize(1);
   }
   
   template<class C>
   ParCell<C>::~ParCell() { }

   template<class C>
   void ParCell<C>::getData(bool sending,int ID,int receivesPosted,int receiveCount,int* blockLengths,MPI_Aint* displacements,MPI_Datatype* types) {
      MPI_Aint baseAddress;
      MPI_Get_address(this,&baseAddress);
      if (ID == 0) {
	 // If this cell is receiving data, allocate memory:
	 if (sending == false) {
	    neighbours.resize(info[0]);
	    neighbourTypes.resize(info[0]);
	    refNumbers.resize(info[1]);
	 }
	 
	 // Get addresses of neighbours,neighbourTypes,refNumbers arrays:
	 MPI_Aint nbrAddress,nbrTypeAddress,refAddress;
	 MPI_Get_address(&(neighbours[0]),&nbrAddress);
	 MPI_Get_address(&(neighbourTypes[0]),&nbrTypeAddress);
	 MPI_Get_address(&(refNumbers[0]),&refAddress);
	 
	 // Write data to output MPI struct:
	 blockLengths[0]  = info[0];
	 displacements[0] = nbrAddress;
	 types[0]         = MPI_Type<CellID>();
	 blockLengths[1]  = info[0];
	 displacements[1] = nbrTypeAddress;
	 types[1]         = MPI_Type<unsigned char>();
	 blockLengths[2]  = info[1];
	 displacements[2] = refAddress;
	 types[2]         = MPI_Type<unsigned char>();
	 
	 // Get rest of cell data from user:
	 userData.getData(sending,ID,receivesPosted,receiveCount,blockLengths+3,displacements+3,types+3);
      } else {
	 // Cells are not migrated, only some cell data is transferred. 
	 // Get data from user:
	 userData.getData(sending,ID,receivesPosted,receiveCount,blockLengths,displacements,types);
      }
   }
   
   template<class C>
   unsigned int ParCell<C>::getDataElements(int ID) const {
      if (ID == 0) return userData.getDataElements(ID) + 3;
      else return userData.getDataElements(ID);
   }
   
   template<class C>
   void ParCell<C>::getMetadata(int ID,int* blockLengths,MPI_Aint* displacements,MPI_Datatype* types) {
      MPI_Aint baseAddress; 
      MPI_Get_address(info,&baseAddress);
      if (ID == 0) {
	 blockLengths[0]  = 2;
	 displacements[0] = baseAddress;
	 types[0] = MPI_Type<unsigned char>();
	 userData.getMetadata(ID,blockLengths+1,displacements+1,types+1);
      } else {
	 userData.getMetadata(ID,blockLengths,displacements,types);
      }
   }
   
   template<class C>
   unsigned int ParCell<C>::getMetadataElements(int ID) const {
      if (ID == 0) return userData.getMetadataElements(ID) + 1;
      return userData.getMetadataElements(ID);
   }
   
   // *************************************************
   // ***** STENCIL TEMPLATE FUNCTION DEFINITIONS *****
   // *************************************************

   /** Default constructor for TypeWrapper. Sets type to MPI_DATATYPE_NULL.*/
   template<class C>
   Stencil<C>::TypeWrapper::TypeWrapper(): type(MPI_DATATYPE_NULL) { }
   
   /** Copy-constructor for TypeWrapper. Makes a copy of MPI datatype with 
    * MPI_Type_dup unless the given datatype is MPI_DATATYPE_NULL.
    * @param tw TypeWrapper to be copied.
    */
   template<class C>
   Stencil<C>::TypeWrapper::TypeWrapper(const Stencil<C>::TypeWrapper& tw) {
      if (tw.type != MPI_DATATYPE_NULL) MPI_Type_dup(tw.type,&type);
      else type = MPI_DATATYPE_NULL;
   }
   
   /** Destructor for TypeWrapper. Frees the MPI datatype with MPI_Type_free 
    * unless the datatype is MPI_DATATYPE_NULL.*/
   template<class C>
   Stencil<C>::TypeWrapper::~TypeWrapper() {
      if (type != MPI_DATATYPE_NULL) MPI_Type_free(&type);
   }
   
   /** Assignment operator for TypeWrapper. Makes a copy of the given MPI datatype 
    * with MPI_Type_dup, unless the given datatype is MPI_DATATYPE_NULL. Also frees 
    * the current datatype with MPI_DATATYPE_NULL if necessary.
    * @param tw TypeWrapper to make copy of.
    * @return Reference to this TypeWrapper.
    */
   template<class C>
   typename Stencil<C>::TypeWrapper& Stencil<C>::TypeWrapper::operator=(const Stencil<C>::TypeWrapper& tw) {
      if (type != MPI_DATATYPE_NULL) MPI_Type_free(&type);
      if (tw.type == MPI_DATATYPE_NULL) type = MPI_DATATYPE_NULL;
      else MPI_Type_dup(tw.type,&type);
      return *this;
   }
   
   template<class C>
   Stencil<C>::Stencil() { }

   template<class C>
   Stencil<C>::~Stencil() {
      // Delete MPI Requests:
      for (typename std::map<int,TypeInfo>::iterator i=typeInfo.begin(); i!=typeInfo.end(); ++i) {
	 delete [] i->second.requests; i->second.requests = NULL;
      }
   }
   
   template<class C>
   bool Stencil<C>::addTransfer(int ID,bool recalculate) {
      if (initialized == false) return false;
      if (typeCaches.find(ID) != typeCaches.end()) return false;
      typeCaches[ID];
      typeInfo[ID].typeVolatile = recalculate;
      typeInfo[ID].requests = NULL;
      typeInfo[ID].started = false;
      calcTypeCache(ID);
      return true;
   }
   
   template<class C>
   bool Stencil<C>::calcLocalUpdateSendsAndReceives() {
      if (initialized == false) return false;
      bool success = true;
      clear();

      // Iterate over all local cells' neighbours. If the neighbour is not in 
      // sentNbrTypeIDs or receivedNbrTypeIDs, skip it.
      // If the neighbour is remote, add the local cell into sends and the neighbour into recvs. 
      // All local cells with one or more remote neighbours are inserted into boundaryCells.
      // Cells with zero remote neighbours are inserted into innerCells instead.
      const std::vector<CellID>& globalIDs = parGrid->getGlobalIDs();
      for (CellID i=0; i<parGrid->getNumberOfLocalCells(); ++i) {
	 unsigned int N_remoteNeighbours = 0;
	 const std::vector<CellID>& nbrIDs = parGrid->getCellNeighbourIDs(i);
	 for (size_t nbr=0; nbr<nbrIDs.size(); ++nbr) {
	    // Check that neighbour exists and that it is not local:
	    const CellID nbrLocalID = nbrIDs[nbr];
	    if (nbrLocalID == parGrid->invalid()) continue;
	    const MPI_processID nbrHost = parGrid->getHosts()[nbrLocalID];
	    if (nbrHost == parGrid->getRank()) continue;
	    const CellID nbrGlobalID = globalIDs[nbrLocalID];
	    
	    // If neighbour type ID is in sentNbrTypeIDs, add a send.
	    if (std::find(sentNbrTypeIDs.begin(),sentNbrTypeIDs.end(),nbr) != sentNbrTypeIDs.end()) {
	       if (stencilType == localToRemoteUpdates)	sends[nbrHost].insert(globalIDs[i]);
	       else sends[nbrHost].insert(nbrGlobalID);
	    }
	    
	    // If neighbour type ID is in receivedNbrTypeIDs, add a receive:
	    if (std::find(receivedNbrTypeIDs.begin(),receivedNbrTypeIDs.end(),nbr) != receivedNbrTypeIDs.end()) {
	       if (stencilType == localToRemoteUpdates) {
		  recvs[nbrHost].insert(nbrGlobalID);
	       } else {
		  recvs[nbrHost].insert(globalIDs[i]);
		  recvCounts[globalIDs[i]].insert(nbrHost);
	       }
	       ++N_remoteNeighbours;
	    }
	 }
	 // Add local cell either into innerCells or boundaryCells:
	 if (N_remoteNeighbours == 0) innerCells.push_back(i);
	 else boundaryCells.push_back(i);
      }
      return success;
   }

   /** Calculate MPI datatype cache for given transfer identifier.
    * @param ID Transfer ID whose cache is to be (re)calculated.
    * @return If true, cache was (re)calculated successfully.*/
   template<class C>
   bool Stencil<C>::calcTypeCache(int ID) {
      typename std::map<int,TypeInfo>::iterator info = typeInfo.find(ID);
      if (info == typeInfo.end()) return false;
      
      int* blockLengths       = NULL;
      MPI_Aint* displacements = NULL;
      MPI_Datatype* types     = NULL;

      // Free old datatypes:
      typename std::map<int,std::map<MPI_processID,TypeCache> >::iterator it=typeCaches.find(ID);
      for (typename std::map<MPI_processID,TypeCache>::iterator jt=it->second.begin(); jt!=it->second.end(); ++jt) {
	 jt->second.sends.clear();
	 jt->second.recvs.clear();
      }
      
      // Delele old MPI_Requests:
      delete [] info->second.requests; info->second.requests = NULL;
      
      // Remove old process entries from typeCache:
      it->second.clear();
      
      // Insert entry for each neighbouring process:
      for (std::map<MPI_processID,std::set<CellID> >::const_iterator i=sends.begin(); i!=sends.end(); ++i) {
	 (it->second)[i->first];
      }
      for (std::map<MPI_processID,std::set<CellID> >::const_iterator i=recvs.begin(); i!=recvs.end(); ++i) {
	 (it->second)[i->first];
      }
      
      // Create MPI datatypes for sending and receiving data:
      info->second.N_receives = 0;
      info->second.N_sends    = 0;
      std::map<CellID,int> receivesPosted;
      for (typename std::map<MPI_processID,TypeCache>::iterator jt=it->second.begin(); jt!=it->second.end(); ++jt) {
	 // Get number of data elements per cell:
	 C dummy;
	 const unsigned int N_dataElements = dummy.getDataElements(ID);
	 
	 // Allocate arrays for MPI datatypes:
	 const size_t N_recvs = recvs.find(jt->first)->second.size() * N_dataElements;
	 const size_t N_sends = sends.find(jt->first)->second.size() * N_dataElements;
	 blockLengths = new int[std::max(N_recvs,N_sends)];
	 displacements = new MPI_Aint[std::max(N_recvs,N_sends)];

	 // Get displacements from cells receiving data:
	 int counter = 0;
	 MPI_Datatype type;
	 for (std::set<CellID>::const_iterator i=recvs[jt->first].begin(); i!=recvs[jt->first].end(); ++i) {
	    const CellID localID = parGrid->getLocalID(*i);
	    const bool sendingData = false;
	    
	    #ifndef NDEBUG
	       if (stencilType == remoteToLocalUpdates) {
		  std::map<CellID,std::set<MPI_processID> >::const_iterator countIt = recvCounts.find(*i);
		  if (countIt == recvCounts.end()) {
		     std::cerr << "(PARGRID) ERROR: recvCounts does not have an entry for local cell with GID #" << *i << std::endl;
		     exit(1);
		  }
	       }
	    #endif
	    
	    switch (stencilType) {
	     case localToRemoteUpdates:
	       (*parGrid)[localID]->getData(sendingData,it->first,-1,-1,blockLengths+counter*N_dataElements,displacements+counter*N_dataElements,&type);
	       break;
	     case remoteToLocalUpdates:
	       (*parGrid)[localID]->getData(sendingData,it->first,receivesPosted[*i],recvCounts[*i].size(),blockLengths+counter*N_dataElements,displacements+counter*N_dataElements,&type);
	       ++receivesPosted[*i];
	       break;
	    }
	    ++counter;
	 }
	    
	 // Create MPI datatype for receiving all data at once from process jt->first:
	 jt->second.recvs.push_back(TypeWrapper());
	 MPI_Type_create_hindexed(N_recvs,blockLengths,displacements,type,&(jt->second.recvs.back().type));
	 MPI_Type_commit(&(jt->second.recvs.back().type));
	 ++info->second.N_receives;
	    
	 // Get displacements from cells sending data:
	 counter = 0;
	 for (std::set<CellID>::const_iterator i=sends[jt->first].begin(); i!=sends[jt->first].end(); ++i) {
	    const CellID localID = parGrid->getLocalID(*i);
	    const bool sendingData = true;
	    const int dummyRecvCount = -1;
	    (*parGrid)[localID]->getData(sendingData,it->first,dummyRecvCount,dummyRecvCount,blockLengths+counter*N_dataElements,displacements+counter*N_dataElements,&type);
	    ++counter;
	 }
	       
	 // Create MPI datatype for sending all data at once to process jt->first.
	 // sum_sends is the total number of datatypes committed (summed over all processes):
	 jt->second.sends.push_back(TypeWrapper());
	 MPI_Type_create_hindexed(N_sends,blockLengths,displacements,type,&(jt->second.sends.back().type));
	 MPI_Type_commit(&(jt->second.sends.back().type));
	 ++info->second.N_sends;
	 
	 delete [] blockLengths; blockLengths = NULL;
	 delete [] displacements; displacements = NULL;
	 delete [] types; types = NULL;
      }
      
      // Allocate enough MPI requests:
      info->second.requests = new MPI_Request[info->second.N_receives+info->second.N_sends];
      return true;
   }

   template<class C>
   void Stencil<C>::clear() {
      boundaryCells.clear();
      innerCells.clear();
      recvs.clear();
      sends.clear();
      recvCounts.clear();
   }
   
   template<class C>
   const std::vector<CellID>& Stencil<C>::getBoundaryCells() const {return boundaryCells;}
   
   template<class C>
   const std::vector<CellID>& Stencil<C>::getInnerCells() const {return innerCells;}
   
   template<class C>
   bool Stencil<C>::initialize(ParGrid<C>& parGrid,StencilType stencilType,const std::vector<unsigned char>& receives) {
      initialized = false;
      this->parGrid = &parGrid;
      this->stencilType = stencilType;
      this->receivedNbrTypeIDs = receives;
      sentNbrTypeIDs.reserve(receives.size());
      
      // Send stencil is the inverse of receive stencil:
      int i_off = 0;
      int j_off = 0;
      int k_off = 0;
      for (size_t i=0; i<receives.size(); ++i) {
	 parGrid.calcNeighbourOffsets(receives[i],i_off,j_off,k_off);
	 i_off *= -1;
	 j_off *= -1;
	 k_off *= -1;
	 sentNbrTypeIDs.push_back(parGrid.calcNeighbourTypeID(i_off,j_off,k_off));
      }

      // Sort vectors containing sent and received neighbour types:
      std::sort(sentNbrTypeIDs.begin(),sentNbrTypeIDs.end());
      std::sort(receivedNbrTypeIDs.begin(),receivedNbrTypeIDs.end());
      
      // Calculate requested stencil:
      initialized = true;
      if (update() == false) initialized = false;
      return initialized;
   }

   /** Remove transfer with given identifier from Stencil.
    * @param ID Identifier of the removed transfer.
    * @return If true, transfer was removed successfully.*/
   template<class C>
   bool Stencil<C>::removeTransfer(int ID) {
      // Check that transfer exists:
      if (initialized == false) return false;
      if (typeCaches.find(ID) == typeCaches.end()) return false;

      // Erase transfer:
      typeCaches.erase(ID);
      typeInfo.erase(ID);
      return true;
   }
   
   template<class C>
   bool Stencil<C>::startTransfer(int ID) {
      typename std::map<int,std::map<MPI_processID,TypeCache> >::iterator it = typeCaches.find(ID);
      typename std::map<int,TypeInfo>::iterator info = typeInfo.find(ID);
      if (it == typeCaches.end()) return false;

      if (info->second.typeVolatile == true) calcTypeCache(ID);
      
      // Post sends and receives:
      unsigned int counter = 0;
      MPI_Request* requests = info->second.requests;
      for (typename std::map<MPI_processID,TypeCache>::iterator proc=it->second.begin(); proc!=it->second.end(); ++proc) {
	 for (size_t i=0; i<proc->second.recvs.size(); ++i) {
	    MPI_Irecv(MPI_BOTTOM,1,proc->second.recvs[i].type,proc->first,proc->first,parGrid->getComm(),requests+counter);
	    ++counter;
	 }
	 for (size_t i=0; i<proc->second.sends.size(); ++i) {
	    MPI_Isend(MPI_BOTTOM,1,proc->second.sends[i].type,proc->first,parGrid->getRank(),parGrid->getComm(),requests+counter);
	    ++counter;
	 }
      }
      info->second.started = true;
      return true;
   }
   
   template<class C>
   bool Stencil<C>::update() {
      if (initialized == false) return false;
      bool success = calcLocalUpdateSendsAndReceives();
      for (typename std::map<int,std::map<MPI_processID,TypeCache> >::iterator it=typeCaches.begin(); it!=typeCaches.end(); ++it) 
	calcTypeCache(it->first);
      return success;
   }
   
   /** Wait for specified transfer to complete. This function waits 
    * for both sends and receives.
    * @param ID Identifier of the transfer.
    * @return If true, transfer has been completed successfully. Value false 
    * may also mean that speciefied transfer has not been initiated.
    */
   template<class C>
   bool Stencil<C>::wait(int ID) {
      typename std::map<int,TypeInfo>::iterator info = typeInfo.find(ID);
      if (info == typeInfo.end()) return false;
      if (info->second.started == false) return false;
      MPI_Waitall(info->second.N_receives+info->second.N_sends,info->second.requests,MPI_STATUSES_IGNORE);
      return true;
   }
   
   // *************************************************
   // ***** PARGRID TEMPLATE FUNCTION DEFINITIONS *****
   // *************************************************
   
   /** Constructor for ParGrid. Initializes Zoltan. Note that MPI_Init must 
    * have been called prior to calling ParGrid constructor.
    * @param hierarchicalLevels Number of hierarchical partitioning levels to use.
    * @param parameters Load balancing parameters for all hierarchical levels.
    * The parameters for each level are given in a list, whose contents are pairs 
    * formed from parameter types and their string values. These lists themselves 
    * are packed into a list, whose first item (list) is used for hierarchical level
    * 0, second item for hierarchical level 1, and so forth.
    */   
   template<class C>
   ParGrid<C>::ParGrid(): initialized(false) {
      recalculateInteriorCells = true;
      recalculateExteriorCells = true;
      N_localCells = 0;
      partitioningCounter = 0;
      
      #ifdef PROFILE
         profZoltanLB  = -1;
         profParGridLB = -1;
         profMPI       = -1;
         profTotalLB   = -1;
      #endif
   }
   
   /** Destructor for ParGrid. Deallocates are user data in cells, i.e. destructor 
    * is called for ParCell::userData for each existing cell. Note that ParGrid 
    * destructor does not call MPI_Finalize.
    */
   template<class C>
     ParGrid<C>::~ParGrid() { }
   
   /** Add a new cell to ParGrid on this process. This function should only 
    * be called when adding initial cells to the grid, i.e. after calling 
    * ParGrid::initialize but before calling initialLoadBalance.
    * In multithreaded mode this function contains readers-writers locks. This 
    * function is not completely thread-safe. If the cell which was just added 
    * is immediately deleted by another thread, then the behaviour will be 
    * undefined. It is safe, however, to call this function simultaneously 
    * with several threads.
    * @return If true, the cell was inserted successfully.
    */
   template<class C>
   bool ParGrid<C>::addCell(CellID cellID,const std::vector<CellID>& nbrIDs,const std::vector<nbrIDtype>& nbrTypes) {
      if (getInitialized() == false) return false;
      
      // Check that the cell doesn't already exist:
      if (global2LocalMap.find(cellID) != global2LocalMap.end()) return false;
      global2LocalMap[cellID] = N_localCells;
      cells.push_back(ParCell<C>());
      hosts.push_back(getRank());
      globalIDs.push_back(cellID);
      
      // Copy cell's neighbours and increase reference count to cell's neighbours:
      cells[N_localCells].info[0] = 27;
      cells[N_localCells].neighbours.resize(27);
      cells[N_localCells].neighbourTypes.resize(27);
      for (size_t n=0; n<27; ++n) {
	 cells[N_localCells].neighbours[n]     = invalid();
	 cells[N_localCells].neighbourTypes[n] = n;
      }
      cells[N_localCells].neighbourFlags = (1 << calcNeighbourTypeID(0,0,0));
      for (size_t n=0; n<nbrIDs.size(); ++n) {
	 cells[N_localCells].neighbourFlags          = (cells[N_localCells].neighbourFlags | (1 << nbrTypes[n]));
	 cells[N_localCells].neighbours[nbrTypes[n]] = nbrIDs[n];
      }
      cells[N_localCells].neighbours[13] = cellID;
      
      ++N_localCells;
      return true;
   }
   
   /** Tell ParGrid that all new cells have been added. 
    * This function contains massive amount of MPI transfer, as 
    * each process must give every other process information on the 
    * cells it has.
    * @return If true, information was shared successfully with other processes.
    */
   template<class C>
   bool ParGrid<C>::addCellFinished() {
      if (getInitialized() == false) return false;
      
      // Insert remote cells and convert global IDs in neighbour lists into local IDs:
      for (size_t i=0; i<N_localCells; ++i) {
	 for (size_t n=0; n<cells[i].neighbours.size(); ++n) {
	    // Skip non-existing neighbours:
	    if (cells[i].neighbours[n] == invalid()) continue;
	    
	    std::map<CellID,CellID>::iterator it = global2LocalMap.find(cells[i].neighbours[n]);
	    if (it != global2LocalMap.end()) {
	       // Neighbour is a local cell. Replace global ID with neighbour's local ID:
	       const CellID localID = it->second;
	       cells[i].neighbours[n] = localID;
	    } else {
	       // Neighbour is a remote cell. Insert a new cell and replace global ID 
	       // with the new local ID:
	       cells.push_back(ParCell<C>());
	       hosts.push_back(MPI_PROC_NULL);
	       globalIDs.push_back(cells[i].neighbours[n]);
	       global2LocalMap[cells[i].neighbours[n]] = cells.size()-1;
	       cells[i].neighbours[n] = cells.size()-1;
	    }
	 }
      }
      
      // Update remote neighbour hosts:
      if (syncCellHosts() == false) {
	 std::cerr << "PARGRID FATAL ERROR: sync cell hosts failed!" << std::endl;
	 return false;
      }
      
      // Calculate neighbour processes:
      nbrProcesses.clear();
      for (size_t i=N_localCells; i<hosts.size(); ++i) {
	 nbrProcesses.insert(hosts[i]);
      }

      #ifndef NDEBUG
         // Check that data from user is ok:
         int successSum = 0;
         int mySuccess = 0;
         if (checkInternalStructures() == false) ++mySuccess;      
         MPI_Allreduce(&mySuccess,&successSum,1,MPI_Type<int>(),MPI_SUM,comm);
         if (successSum > 0) return false;
      #endif
      
      // Add default stencil (all neighbours exchange data):
      std::vector<unsigned char> nbrTypeIDs(27);
      for (size_t i=0; i<27; ++i) {
	 if (i == 13) continue;
	 nbrTypeIDs[i] = i;
      }
      stencils[0].initialize(*this,localToRemoteUpdates,nbrTypeIDs);

      // Invalidate all internal variables that depend on cell partitioning:
      invalidate();
      return true;
   }

   /** Create a new Stencil for MPI transfers.
    * @param stencilType Type of Stencil, one of the values defined in enum StencilType.
    * @param recvNbrTypeIDs Neighbour type IDs that identify the neighbours whom to receive data from.
    * @return If greater than zero the Stencil was created successfully.*/
   template<class C>
   int ParGrid<C>::addStencil(pargrid::StencilType stencilType,const std::vector<unsigned char>& recvNbrTypeIDs) {
      int currentSize = stencils.size();
      if (stencils[currentSize].initialize(*this,stencilType,recvNbrTypeIDs) == false) {
	 stencils.erase(currentSize);
	 currentSize = -1;
      }
      return currentSize;
   }
   
   template<class C>
   bool ParGrid<C>::addTransfer(unsigned stencil,int identifier,bool recalculate) {
      if (getInitialized() == false) return false;
      typename std::map<unsigned int,Stencil<C> >::iterator it = stencils.find(stencil);
      if (it == stencils.end()) return false;
      return it->second.addTransfer(identifier,recalculate);
   }
   
   /** Balance load based on criteria given in ParGrid::initialize. Load balancing 
    * invalidates lists of cells etc. stored outside of ParGrid.
    * @return If true, load was balanced successfully.
    */
   template<class C>
   bool ParGrid<C>::balanceLoad() {
      #ifdef PROFILE
         profile::start("ParGrid load balance",profTotalLB);
      #endif
      bool success = true;
      
      // ********************************************************
      // ***** STEP 1: REQUEST NEW PARTITIONING FROM ZOLTAN *****
      // ********************************************************
      
      // Request load balance from Zoltan, and get cells which should be imported and exported.
      // NOTE that import/export lists may contain cells that already are on this process, at
      // least with RANDOM load balancing method!
      #ifdef PROFILE
         profile::start("Zoltan LB",profZoltanLB);
      #endif
      int changes,N_globalIDs,N_localIDs,N_import,N_export;
      int* importProcesses = NULL;
      int* importParts     = NULL;
      int* exportProcesses = NULL;
      int* exportParts     = NULL;
      ZOLTAN_ID_PTR importGlobalIDs = NULL;
      ZOLTAN_ID_PTR importLocalIDs  = NULL;
      ZOLTAN_ID_PTR exportGlobalIDs = NULL;
      ZOLTAN_ID_PTR exportLocalIDs  = NULL;
      if (zoltan->LB_Partition(changes,N_globalIDs,N_localIDs,N_import,importGlobalIDs,importLocalIDs,
			       importProcesses,importParts,N_export,exportGlobalIDs,exportLocalIDs,
			       exportProcesses,exportParts) != ZOLTAN_OK) {
	 std::cerr << "ParGrid FATAL ERROR: Zoltan failed on load balancing!" << std::endl << std::flush;
	 zoltan->LB_Free_Part(&importGlobalIDs,&importLocalIDs,&importProcesses,&importParts);
	 zoltan->LB_Free_Part(&exportGlobalIDs,&exportLocalIDs,&exportProcesses,&exportParts);
	 success = false;
	 #ifdef PROFILE
	    profile::stop();
	    profile::stop();
	 #endif
	 return success;
      }
      #ifdef PROFILE
         profile::stop();
         profile::start("ParGrid LB",profParGridLB);
      #endif
      
      // Allocate memory for new cell array and start to receive imported cells.
      // Note. N_newLocalCells is correct even if N_import and N_export contain 
      // transfers from process A to process A.
      const double totalToLocalRatio = (1.0*cells.size()) / (N_localCells+1.0e-20);
      const CellID N_newLocalCells = N_localCells + N_import - N_export;
      const CellID newCapacity = static_cast<CellID>(std::ceil((totalToLocalRatio+0.1)*N_newLocalCells));

      // Calculate index into newCells where imported cells from other processes are 
      // inserted. N_realExports != N_export because export list may contain 
      // exports from process A to process A, i.e. not all cells are migrated.
      int N_realExports = 0;
      for (int i=0; i<N_export; ++i) if (exportProcesses[i] != getRank()) ++N_realExports;
      const CellID newLocalsBegin = N_localCells - N_realExports;
      
      std::vector<ParCell<C> > newCells;
      std::vector<MPI_processID> newHosts;
      std::map<CellID,CellID> newGlobal2LocalMap;
      std::vector<CellID> newGlobalIDs;      
      newCells.reserve(newCapacity);
      newHosts.reserve(newCapacity);
      newGlobalIDs.reserve(newCapacity);
      newCells.resize(N_newLocalCells);

      // ********************************************************** //
      // ***** EXCHANGE IMPORTED AND EXPORTED CELLS' METADATA ***** //
      // ********************************************************** //
      
      // Count the number of cells exported per neighbouring process, and number of cells 
      // imported per neighbouring process. All data needs to be sent (and received) with a 
      // single MPI call per remote neighbour, otherwise unexpected message buffers may run out!
      // Note: importsPerProcess and exportsPerProcess may contain transfers from process A to process A.
      std::map<MPI_processID,unsigned int> exportsPerProcess;
      std::map<MPI_processID,unsigned int> importsPerProcess;
      for (int i=0; i<N_import; ++i) ++importsPerProcess[importProcesses[i]];
      for (int i=0; i<N_export; ++i) ++exportsPerProcess[exportProcesses[i]];

      // Erase imports/exports from process A to process A:
      exportsPerProcess.erase(getRank());
      importsPerProcess.erase(getRank());
      
      size_t counter = 0;
      
      // Allocate enough MPI Requests for sends and receives used in cell migration:
      size_t recvSize = std::max(2*nbrProcesses.size(),static_cast<size_t>(N_import));
      size_t sendSize = std::max(2*nbrProcesses.size(),static_cast<size_t>(N_export));
      recvRequests.resize(recvSize);
      sendRequests.resize(sendSize);

      // Get the number of metadata elements per cell:
      ParCell<C> dummy;
      unsigned int N_metadataElements = dummy.getMetadataElements(0);
      
      // Create temporary arrays for metadata incoming from each importing process:
      std::vector<MPI_Datatype*> datatypes(importsPerProcess.size());
      std::vector<int*> blockLengths(importsPerProcess.size());
      std::vector<MPI_Aint*> displacements(importsPerProcess.size());
      std::map<MPI_processID,size_t> processIndices;
      std::vector<size_t> indices(importsPerProcess.size());
      counter = 0;
      for (std::map<MPI_processID,unsigned int>::const_iterator it=importsPerProcess.begin(); it!=importsPerProcess.end(); ++it) {
	 datatypes[counter]        = new MPI_Datatype[N_metadataElements*it->second];
	 blockLengths[counter]     = new int[N_metadataElements*it->second];
	 displacements[counter]    = new MPI_Aint[N_metadataElements*it->second];
	 processIndices[it->first] = counter;
	 indices[counter]          = 0;
	 ++counter;
      }      
      
      // Get metadata for imported cells. In practice we fetch addresses where
      // incoming metadata is to be written:
      counter = 0;
      for (int i=0; i<N_import; ++i) {
	 if (importProcesses[i] == getRank()) continue;
	 const int identifier = 0;
	 const unsigned int procIndex = processIndices[importProcesses[i]];
	 const unsigned int arrayIndex = indices[procIndex];
	 newCells[newLocalsBegin+counter].getMetadata(identifier,blockLengths[procIndex]+arrayIndex,displacements[procIndex]+arrayIndex,datatypes[procIndex]+arrayIndex);
	 indices[procIndex] += N_metadataElements;
	 ++counter;
      }
      
      // Create an MPI struct containing all metadata received from importing process it->first and post receives:
      MPI_Datatype MPIdataType;
      counter = 0;
      for (std::map<MPI_processID,unsigned int>::const_iterator it=importsPerProcess.begin(); it!=importsPerProcess.end(); ++it) {
	 const MPI_processID source = it->first;
	 const int tag              = it->first;
	 MPI_Type_create_struct(N_metadataElements*it->second,blockLengths[counter],displacements[counter],datatypes[counter],&MPIdataType);
	 MPI_Type_commit(&MPIdataType);
	 MPI_Irecv(MPI_BOTTOM,1,MPIdataType,source,tag,comm,&(recvRequests[counter]));
	 MPI_Type_free(&MPIdataType);
	 ++counter;
      }
      
      // Deallocate arrays used for imported metadata:
      for (size_t i=0; i<importsPerProcess.size(); ++i) {
	 delete [] datatypes[i]; datatypes[i] = NULL;
	 delete [] blockLengths[i]; blockLengths[i] = NULL;
	 delete [] displacements[i]; displacements[i] = NULL;
      }
      
      // Allocate arrays for sending metadata to export processes:
      datatypes.resize(exportsPerProcess.size());
      blockLengths.resize(exportsPerProcess.size());
      displacements.resize(exportsPerProcess.size());
      indices.resize(exportsPerProcess.size());
      processIndices.clear();
      counter = 0;
      for (std::map<MPI_processID,unsigned int>::const_iterator it=exportsPerProcess.begin(); it!=exportsPerProcess.end(); ++it) {
	 datatypes[counter]        = new MPI_Datatype[N_metadataElements*it->second];
	 blockLengths[counter]     = new int[N_metadataElements*it->second];
	 displacements[counter]    = new MPI_Aint[N_metadataElements*it->second];
	 processIndices[it->first] = counter;
	 indices[counter] = 0;
	 ++counter;
      }
      
      // Fetch metadata to send to export processes:
      for (int i=0; i<N_export; ++i) {
	 if (exportProcesses[i] == getRank()) continue;
	 
	 const int identifier = 0;
	 const size_t procIndex = processIndices[exportProcesses[i]];
	 const unsigned int arrayIndex = indices[procIndex];
	 cells[exportLocalIDs[i]].getMetadata(identifier,blockLengths[procIndex]+arrayIndex,displacements[procIndex]+arrayIndex,datatypes[procIndex]+arrayIndex);
	 indices[procIndex] += N_metadataElements;
      }
      
      // Create an MPI struct containing all metadata sent to export process it->first and post send:
      counter = 0;
      for (std::map<MPI_processID,unsigned int>::const_iterator it=exportsPerProcess.begin(); it!=exportsPerProcess.end(); ++it) {
	 const MPI_processID dest = it->first;
	 const int tag            = myrank;
	 MPI_Type_create_struct(N_metadataElements*it->second,blockLengths[counter],displacements[counter],datatypes[counter],&MPIdataType);
	 MPI_Type_commit(&MPIdataType);
	 MPI_Isend(MPI_BOTTOM,1,MPIdataType,dest,tag,comm,&(sendRequests[counter]));
	 MPI_Type_free(&MPIdataType);
	 ++counter;
      }
      
      // Deallocate arrays used for MPI datatypes in cell exports:
      for (size_t i=0; i<exportsPerProcess.size(); ++i) {
	 delete [] datatypes[i]; datatypes[i] = NULL;
	 delete [] blockLengths[i]; blockLengths[i] = NULL;
	 delete [] displacements[i]; displacements[i] = NULL;
      }

      // Set new host for all exported cells, it should not matter 
      // if cells are exported from process A to process A here:
      for (int c=0; c<N_export; ++c) {
	 hosts[exportLocalIDs[c]] = exportProcesses[c];
      }

      #ifdef PROFILE
         profile::start("MPI waits",profMPI);
      #endif
      // Wait for metadata sends & receives to complete:
      MPI_Waitall(importsPerProcess.size(),&(recvRequests[0]),MPI_STATUSES_IGNORE);
      MPI_Waitall(exportsPerProcess.size(),&(sendRequests[0]),MPI_STATUSES_IGNORE);
      #ifdef PROFILE
         profile::stop();
      #endif
      
      // ****************************************************** //
      // ***** EXCHANGE IMPORTED AND EXPORTED CELLS' DATA ***** //
      // ****************************************************** //

      const unsigned int N_dataElements = dummy.getDataElements(0);

      // Swap all cells' neighbour IDs to global IDs:
      for (CellID c=0; c<N_localCells; ++c) {
	 for (size_t n=0; n<cells[c].neighbours.size(); ++n) {
	    if (cells[c].neighbours[n] == invalid()) continue;
	    cells[c].neighbours[n] = globalIDs[cells[c].neighbours[n]];
	 }
      }
      
      
      // Allocate arrays for MPI datatypes describing imported cell data:
      datatypes.resize(importsPerProcess.size());
      blockLengths.resize(importsPerProcess.size());
      displacements.resize(importsPerProcess.size());
      processIndices.clear();
      indices.resize(importsPerProcess.size());
      counter = 0;
      for (std::map<MPI_processID,unsigned int>::const_iterator it=importsPerProcess.begin(); it!=importsPerProcess.end(); ++it) {
	 datatypes[counter]        = new MPI_Datatype[N_dataElements*it->second];
	 blockLengths[counter]     = new int[N_dataElements*it->second];
	 displacements[counter]    = new MPI_Aint[N_dataElements*it->second];
	 processIndices[it->first] = counter;
	 indices[counter]          = 0;
	 ++counter;
      }
      
      // Fetch data to arrays describing imported data:
      counter = 0;
      for (int i=0; i<N_import; ++i) {
	 if (importProcesses[i] == getRank()) continue;
	 
	 const bool sending = false;
	 const int identifier = 0;
	 const int dummyRecvCount = -1;
	 const unsigned int procIndex = processIndices[importProcesses[i]];
	 const unsigned int arrayIndex = indices[procIndex];
	 newCells[newLocalsBegin+counter].getData(sending,identifier,0,dummyRecvCount,blockLengths[procIndex]+arrayIndex,displacements[procIndex]+arrayIndex,datatypes[procIndex]+arrayIndex);
	 indices[procIndex] += N_dataElements;
	 ++counter;
      }
      
      // Create an MPI struct containing all data received from importing process it->first and post receives:
      counter = 0;
      for (std::map<MPI_processID,unsigned int>::const_iterator it=importsPerProcess.begin(); it!=importsPerProcess.end(); ++it) {
	 const MPI_processID source = it->first;
	 const int tag              = it->first;
	 MPI_Type_create_struct(N_dataElements*it->second,blockLengths[counter],displacements[counter],datatypes[counter],&MPIdataType);
	 MPI_Type_commit(&MPIdataType);
	 MPI_Irecv(MPI_BOTTOM,1,MPIdataType,source,tag,comm,&(recvRequests[counter]));
	 MPI_Type_free(&MPIdataType);
	 ++counter;
      }
      
      // Deallocate arrays:
      for (size_t i=0; i<importsPerProcess.size(); ++i) {
	 delete [] datatypes[i]; datatypes[i] = NULL;
	 delete [] blockLengths[i]; blockLengths[i] = NULL;
	 delete [] displacements[i]; displacements[i] = NULL;
      }
      
      // Allocate arrays containing MPI datatypes for exported cell data:
      datatypes.resize(exportsPerProcess.size());
      blockLengths.resize(exportsPerProcess.size());
      displacements.resize(exportsPerProcess.size());
      indices.resize(exportsPerProcess.size());
      processIndices.clear();
      counter = 0;
      for (std::map<MPI_processID,unsigned int>::const_iterator it=exportsPerProcess.begin(); it!=exportsPerProcess.end(); ++it) {
	 datatypes[counter]        = new MPI_Datatype[N_dataElements*it->second];
	 blockLengths[counter]     = new int[N_dataElements*it->second];
	 displacements[counter]    = new MPI_Aint[N_dataElements*it->second];
	 processIndices[it->first] = counter;
	 indices[counter] = 0;
	 ++counter;
      }
      
      // Fetch data to arrays desbribing exported cell data:
      for (int i=0; i<N_export; ++i) {
	 if (exportProcesses[i] == getRank()) continue;
	 
	 const bool sending = true;
	 const int identifier = 0;
	 const int dummyRecvCount = -1;
	 const size_t procIndex = processIndices[exportProcesses[i]];
	 const unsigned int arrayIndex = indices[procIndex];
	 cells[exportLocalIDs[i]].getData(sending,identifier,dummyRecvCount,dummyRecvCount,blockLengths[procIndex]+arrayIndex,displacements[procIndex]+arrayIndex,datatypes[procIndex]+arrayIndex);
	 indices[procIndex] += N_dataElements;
	 ++counter;
      }
      
      // Create an MPI struct containing all data sent to export process it->first and post send:
      counter = 0;
      for (std::map<MPI_processID,unsigned int>::const_iterator it=exportsPerProcess.begin(); it!=exportsPerProcess.end(); ++it) {
	 const MPI_processID dest = it->first;
	 const int tag            = myrank;
	 MPI_Type_create_struct(N_dataElements*it->second,blockLengths[counter],displacements[counter],datatypes[counter],&MPIdataType);
	 MPI_Type_commit(&MPIdataType);
	 MPI_Isend(MPI_BOTTOM,1,MPIdataType,dest,tag,comm,&(sendRequests[counter]));
	 MPI_Type_free(&MPIdataType);
	 ++counter;
      }
      
      // Deallocate arrays used for MPI datatypes in cell exports:
      for (size_t i=0; i<exportsPerProcess.size(); ++i) {
	 delete [] datatypes[i]; datatypes[i] = NULL;
	 delete [] blockLengths[i]; blockLengths[i] = NULL;
	 delete [] displacements[i]; displacements[i] = NULL;
      }
      
      // Copy non-exported local cells to newCells:
      counter = 0;
      for (CellID c=0; c<N_localCells; ++c) {
	 if (hosts[c] != getRank()) continue;
	 newCells[counter] = cells[c];
	 newGlobalIDs.push_back(globalIDs[c]);
	 newHosts.push_back(getRank());
	 newGlobal2LocalMap[globalIDs[c]] = counter;
	 ++counter;
      }

      // Set host, global ID, and global2Local entries for imported cells:
      counter = 0;
      for (int i=0; i<N_import; ++i) {
	 if (importProcesses[i] == getRank()) continue;
	 
	 newHosts.push_back(getRank());
	 newGlobalIDs.push_back(importGlobalIDs[i]);
	 newGlobal2LocalMap[importGlobalIDs[i]] = newLocalsBegin + counter;
	 ++counter;
      }

      #ifdef PROFILE
         profile::start("MPI Waits",profMPI);
      #endif
      // Wait for cell data recvs & sends to complete:
      MPI_Waitall(importsPerProcess.size(),&(recvRequests[0]),MPI_STATUSES_IGNORE);
      MPI_Waitall(exportsPerProcess.size(),&(sendRequests[0]),MPI_STATUSES_IGNORE);
      #ifdef PROFILE
         profile::stop();
      #endif
      
      // ************************************************************ //
      // ***** INSERT REMOTE CELLS AND UPDATE REMOTE CELL HOSTS ***** //
      // ************************************************************ //
      
      // Iterate over new cell list and replace neighbour GIDs with LIDs.
      // Also calculate neighbourFlags for all local cells:
      for (CellID c=0; c<N_newLocalCells; ++c) {
	 newCells[c].neighbourFlags = (1 << calcNeighbourTypeID(0,0,0));
	 
	 std::map<CellID,CellID>::const_iterator it;
	 for (size_t n=0; n<newCells[c].neighbours.size(); ++n) {
	    const CellID nbrGID = newCells[c].neighbours[n];
	    if (nbrGID == invalid()) continue;
	    
	    CellID nbrLID = invalid();
	    it = newGlobal2LocalMap.find(nbrGID);
	    if (it != newGlobal2LocalMap.end()) {
	       nbrLID = it->second;
	    } else {
	       // Insert remote cell to newCells:
	       nbrLID = newCells.size();
	       newCells.push_back(ParCell<C>());
	       newGlobalIDs.push_back(nbrGID);
	       newGlobal2LocalMap[nbrGID] = nbrLID;
	       newHosts.push_back(MPI_PROC_NULL);
	       
	       // If remote cell already exists on this process, copy host ID from hosts:
	       it = global2LocalMap.find(nbrGID);
	       if (it != global2LocalMap.end()) {
		  const CellID nbrOldLID = it->second;
		  newHosts[nbrLID] = hosts[nbrOldLID];
	       }
	    }

	    newCells[c].neighbours[n]  = nbrLID;
	    newCells[c].neighbourFlags = (newCells[c].neighbourFlags | (1 << newCells[c].neighbourTypes[n]));
	 }
      }

      // First host update pass:
      // Send list of exported cells and their new hosts to every remote process:
      int* neighbourChanges                      = new int[nbrProcesses.size()];              // Number of exports nbrs. process has
      ZOLTAN_ID_TYPE** neighbourMigratingCellIDs = new ZOLTAN_ID_TYPE* [nbrProcesses.size()]; // Global IDs of cells nbr. process is exporting
      MPI_processID** neighbourMigratingHosts    = new MPI_processID* [nbrProcesses.size()];  // New hosts for cells nbr. process exports
      for (size_t i=0; i<nbrProcesses.size(); ++i) {
	 neighbourMigratingCellIDs[i] = NULL;
	 neighbourMigratingHosts[i]   = NULL;
      }
      
      // Send the number of cells this process is exporting to all neighbouring process,
      // and receive the number of exported cells per neighbouring process:
      counter = 0;
      for (std::set<MPI_processID>::const_iterator it=nbrProcesses.begin(); it!=nbrProcesses.end(); ++it) {
	 MPI_Irecv(&(neighbourChanges[counter]),1,MPI_INT,*it,   *it,comm,&(recvRequests[counter]));
	 MPI_Isend(&N_export,                   1,MPI_INT,*it,myrank,comm,&(sendRequests[counter]));
	 ++counter;
      }
      
      // Wait for information to arrive from neighbours:
      #ifdef PROFILE
         profile::start("MPI Waits",profMPI);
      #endif
      MPI_Waitall(nbrProcesses.size(),&(recvRequests[0]),MPI_STATUSES_IGNORE);
      MPI_Waitall(nbrProcesses.size(),&(sendRequests[0]),MPI_STATUSES_IGNORE);
      #ifdef PROFILE
         profile::stop();
      #endif
      
      // Allocate arrays for receiving migrating cell IDs and new hosts
      // from neighbouring processes. Exchange data with neighbours:
      counter = 0;
      for (std::set<MPI_processID>::const_iterator it=nbrProcesses.begin(); it!=nbrProcesses.end(); ++it) {
	 neighbourMigratingCellIDs[counter] = new ZOLTAN_ID_TYPE[neighbourChanges[counter]];
	 neighbourMigratingHosts[counter]   = new MPI_processID[neighbourChanges[counter]];
	 
	 MPI_Irecv(neighbourMigratingCellIDs[counter],neighbourChanges[counter],MPI_Type<ZOLTAN_ID_TYPE>(),*it,*it,comm,&(recvRequests[2*counter+0]));
	 MPI_Irecv(neighbourMigratingHosts[counter]  ,neighbourChanges[counter],MPI_Type<MPI_processID>() ,*it,*it,comm,&(recvRequests[2*counter+1]));
	 MPI_Isend(exportGlobalIDs,N_export,MPI_Type<ZOLTAN_ID_TYPE>(),*it,myrank,comm,&(sendRequests[2*counter+0]));
	 MPI_Isend(exportProcesses,N_export,MPI_Type<MPI_processID>() ,*it,myrank,comm,&(sendRequests[2*counter+1]));
	 ++counter;
      }
      
      #ifdef PROFILE
         profile::start("MPI Waits",profMPI);
      #endif
      MPI_Waitall(2*nbrProcesses.size(),&(recvRequests[0]),MPI_STATUSES_IGNORE);
      MPI_Waitall(2*nbrProcesses.size(),&(sendRequests[0]),MPI_STATUSES_IGNORE);
      #ifdef PROFILE
         profile::stop();
      #endif
      
      // Update hosts, this should work even if export list from process A has exports to process A:
      counter = 0;
      for (std::set<MPI_processID>::const_iterator it=nbrProcesses.begin(); it!=nbrProcesses.end(); ++it) {
	 for (int i=0; i<neighbourChanges[counter]; ++i) {
	    // Update entry in hosts (needed for second host update pass below):
	    std::map<CellID,CellID>::const_iterator it = global2LocalMap.find(neighbourMigratingCellIDs[counter][i]);
	    if (it != global2LocalMap.end()) hosts[it->second] = neighbourMigratingHosts[counter][i];
	    
	    // Update entry in newHosts:
	    it = newGlobal2LocalMap.find(neighbourMigratingCellIDs[counter][i]);
	    if (it == newGlobal2LocalMap.end()) continue;
	    
	    // Skip cells imported to this process:
	    const CellID nbrLID = it->second;
	    if (nbrLID < N_newLocalCells) continue;	    
	    newHosts[nbrLID] = neighbourMigratingHosts[counter][i];
	 }
	 ++counter;
      }
      
      // Deallocate memory:
      delete [] neighbourChanges; neighbourChanges = NULL;
      for (size_t i=0; i<nbrProcesses.size(); ++i) {
	 delete neighbourMigratingCellIDs[i]; neighbourMigratingCellIDs[i] = NULL;
	 delete neighbourMigratingHosts[i]; neighbourMigratingHosts[i] = NULL;
      }
      delete [] neighbourMigratingCellIDs; neighbourMigratingCellIDs = NULL;
      delete [] neighbourMigratingHosts; neighbourMigratingHosts = NULL;
      
      // Calculate unique import / export processes:
      std::set<MPI_processID> importProcs;
      std::set<MPI_processID> exportProcs;
      for (int i=0; i<N_import; ++i) importProcs.insert(importProcesses[i]);
      for (int i=0; i<N_export; ++i) exportProcs.insert(exportProcesses[i]);
      importProcs.erase(getRank());
      exportProcs.erase(getRank());
      
      // Second host update pass.
      // Go over exported cells' neighbours. If neighbour is not in exported cell's 
      // new host, add an entry to hostUpdates:
      std::map<MPI_processID,std::set<std::pair<CellID,MPI_processID> > > hostUpdates;
      for (int i=0; i<N_export; ++i) {
	 if (exportProcesses[i] == getRank()) continue;

	 const CellID exportLID = exportLocalIDs[i];
	 const MPI_processID newHost = hosts[exportLID];
	 for (size_t n=0; n<cells[exportLID].neighbours.size(); ++n) {
	    const CellID nbrGID = cells[exportLID].neighbours[n];
	    if (nbrGID == invalid()) continue;
	    std::map<CellID,CellID>::const_iterator it=global2LocalMap.find(nbrGID);
	    #ifndef NDEBUG
	       if (it == global2LocalMap.end()) {
		  std::cerr << "(PARGRID) ERROR: Nbr cell with GID " << nbrGID << " was not found from global2LocalMap" << std::endl;
		  exit(1);
	       }
	    #endif
	    const MPI_processID nbrHost = hosts[it->second];
	    if (nbrHost != newHost) hostUpdates[newHost].insert(std::make_pair(nbrGID,nbrHost));
	 }
      }
      
      // Recv hostUpdates list from each process in importProcs:
      counter = 0;
      size_t* incomingUpdates = new size_t[importProcs.size()];
      for (std::set<MPI_processID>::const_iterator it=importProcs.begin(); it!=importProcs.end(); ++it) {
	 MPI_Irecv(incomingUpdates+counter,1,MPI_Type<size_t>(),*it,*it,comm,&(recvRequests[counter]));
	 ++counter;
      }
      
      counter = 0;
      size_t* outgoingUpdates = new size_t[exportProcs.size()];
      for (std::set<MPI_processID>::const_iterator it=exportProcs.begin(); it!=exportProcs.end(); ++it) {
	 // hostUpdates[*it].size() below inserts an entry into hostUpdates for process *it if one did not exist:
	 outgoingUpdates[counter] = hostUpdates[*it].size();
	 MPI_Isend(outgoingUpdates+counter,1,MPI_Type<size_t>(),*it,myrank,comm,&(sendRequests[counter]));
	 ++counter;
      }
      
      // Wait for number of host updates sends and recvs to complete:
      #ifdef PROFILE
         profile::start("MPI Waits",profMPI);
      #endif
      MPI_Waitall(importProcs.size(),&(recvRequests[0]),MPI_STATUSES_IGNORE);
      MPI_Waitall(exportProcs.size(),&(sendRequests[0]),MPI_STATUSES_IGNORE);
      #ifdef PROFILE
         profile::stop();
      #endif
      
      // Allocate buffers for sending and receiving host updates:
      if (sendRequests.size() < 2*exportProcs.size()) sendRequests.resize(2*exportProcs.size());
      if (recvRequests.size() < 2*importProcs.size()) recvRequests.resize(2*importProcs.size());
      CellID** incomingCellIDs      = new CellID* [importProcs.size()];
      CellID** outgoingCellIDs      = new CellID* [exportProcs.size()];
      MPI_processID** incomingHosts = new MPI_processID* [importProcs.size()];
      MPI_processID** outgoingHosts = new MPI_processID* [exportProcs.size()];
      for (size_t i=0; i<importProcs.size(); ++i) {
	 incomingCellIDs[i] = NULL;
	 incomingHosts[i] = NULL;
      }
      for (size_t i=0; i<exportProcs.size(); ++i) {
	 outgoingCellIDs[i] = NULL;
	 outgoingHosts[i] = NULL;
      }

      // Copy data to send buffers:
      counter = 0;
      for (std::set<MPI_processID>::const_iterator it=exportProcs.begin(); it!=exportProcs.end(); ++it) {
	 outgoingCellIDs[counter] = new CellID[outgoingUpdates[counter]];
	 outgoingHosts[counter]   = new MPI_processID[outgoingUpdates[counter]];
	 
	 unsigned int i = 0;
	 for (std::set<std::pair<CellID,MPI_processID> >::const_iterator jt=hostUpdates[*it].begin(); jt!=hostUpdates[*it].end(); ++jt) {
	    outgoingCellIDs[counter][i] = jt->first;
	    outgoingHosts[counter][i]   = jt->second;
	    ++i;
	 }
	 ++counter;
      }
      
      // Allocate buffers for incoming cell updates:
      for (size_t i=0; i<importProcs.size(); ++i) {
	 incomingCellIDs[i] = new CellID[incomingUpdates[i]];
	 incomingHosts[i]   = new MPI_processID[incomingUpdates[i]];
      }
      
      // Receive host updates from neighbouring processes:
      counter = 0;
      for (std::set<MPI_processID>::const_iterator it=importProcs.begin(); it!=importProcs.end(); ++it) {
	 MPI_Irecv(incomingCellIDs[counter],incomingUpdates[counter],MPI_Type<CellID>()       ,*it,*it,comm,&(recvRequests[2*counter+0]));
	 MPI_Irecv(incomingHosts[counter]  ,incomingUpdates[counter],MPI_Type<MPI_processID>(),*it,*it,comm,&(recvRequests[2*counter+1]));
	 ++counter;
      }
      
      // Send this process' update list to neighbouring processes:
      counter = 0;
      for (std::set<MPI_processID>::const_iterator it=exportProcs.begin(); it!=exportProcs.end(); ++it) {
	 MPI_Isend(outgoingCellIDs[counter],outgoingUpdates[counter],MPI_Type<CellID>()       ,*it,myrank,comm,&(sendRequests[2*counter+0]));
	 MPI_Isend(outgoingHosts[counter]  ,outgoingUpdates[counter],MPI_Type<MPI_processID>(),*it,myrank,comm,&(sendRequests[2*counter+1]));
	 ++counter;
      }
      
      // Wait for host updates to complete:
      #ifdef PROFILE
         profile::start("MPI Waits",profMPI);
      #endif
      MPI_Waitall(2*importProcs.size(),&(recvRequests[0]),MPI_STATUSES_IGNORE);
      MPI_Waitall(2*exportProcs.size(),&(sendRequests[0]),MPI_STATUSES_IGNORE);
      #ifdef PROFILE
         profile::stop();
      #endif

      // Update hosts based on received information:
      counter = 0;
      for (std::set<MPI_processID>::const_iterator it=importProcs.begin(); it!=importProcs.end(); ++it) {
	 for (size_t i=0; i<incomingUpdates[counter]; ++i) {
	    std::map<CellID,CellID>::const_iterator it = newGlobal2LocalMap.find(incomingCellIDs[counter][i]);
	    if (it == newGlobal2LocalMap.end()) continue;
	    const CellID LID = it->second;
	    newHosts[LID] = incomingHosts[counter][i];
	 }
	 ++counter;
      }

      // Deallocate arrays used in host updates:
      delete [] incomingUpdates; incomingUpdates = NULL;
      delete [] outgoingUpdates; outgoingUpdates = NULL;
      for (size_t i=0; i<importProcs.size(); ++i) {
	 delete [] incomingCellIDs[i]; incomingCellIDs[i] = NULL;
	 delete [] incomingHosts[i]; incomingHosts[i] = NULL;
      }
      delete [] incomingHosts; incomingHosts = NULL;
      delete [] incomingCellIDs; incomingCellIDs = NULL;
      for (size_t i=0; i<exportProcs.size(); ++i) {
	 delete [] outgoingCellIDs[i]; outgoingCellIDs[i] = NULL;
	 delete [] outgoingHosts[i]; outgoingHosts[i] = NULL;
      }
      delete [] outgoingHosts; outgoingHosts = NULL;
      delete [] outgoingCellIDs; outgoingCellIDs = NULL;

      // Deallocate Zoltan arrays:
      zoltan->LB_Free_Part(&importGlobalIDs,&importLocalIDs,&importProcesses,&importParts);
      zoltan->LB_Free_Part(&exportGlobalIDs,&exportLocalIDs,&exportProcesses,&exportParts);

      // Swap cell data arrays:
      cells.swap(newCells);
      hosts.swap(newHosts);
      globalIDs.swap(newGlobalIDs);
      global2LocalMap.swap(newGlobal2LocalMap);
      N_localCells = N_newLocalCells;
      
      // Recalculate and/or invalidate all other internal data that depends on partitioning:
      nbrProcesses.clear();
      for (CellID c=N_localCells; c<hosts.size(); ++c) {
	 nbrProcesses.insert(hosts[c]);
      }
      invalidate();
      
      #ifndef NDEBUG
         checkInternalStructures();
      #endif

      ++partitioningCounter;
      
      #ifdef PROFILE
         profile::stop();
         profile::stop();
      #endif
      return success;
   }
   
   /** Synchronize MPI processes in the communicator ParGrid is using.*/
   template<class C>
   void ParGrid<C>::barrier(int threadID) const {
      MPI_Barrier(comm);
   }
   
   template<class C>
   void ParGrid<C>::calcNeighbourOffsets(unsigned char nbrTypeID,int& i_off,int& j_off,int& k_off) const {
      int tmp = nbrTypeID;
      k_off = tmp / 9;
      tmp -= k_off*9;
      j_off = tmp / 3;
      tmp -= j_off*3;
      i_off = tmp;
      --i_off;
      --j_off;
      --k_off;
   }
   
   /** Calculate neighbour type ID corresponding to the given cell offsets.
    * Valid offset values are [-1,0,+1].
    * @param i_off Cell offset in first coordinate direction.
    * @param j_off Cell offset in second coordinate direction.
    * @param k_off Cell offset in third coordinate direction.
    * @return Calculated neighbour type ID.
    */
   template<class C>
   unsigned char ParGrid<C>::calcNeighbourTypeID(int i_off,int j_off,int k_off) const {
      return (k_off+1)*9 + (j_off+1)*3 + i_off+1;
   }

   /** Debugging function, checks ParGrid internal structures for correctness.
    * @return If true, everything is ok.
    */
   template<class C>
   bool ParGrid<C>::checkInternalStructures() const {
      if (getInitialized() == false) return false;
      bool success = true;
      std::map<CellID,int> tmpReferences;
      std::set<CellID> tmpRemoteCells;
      
      // Count neighbour references, and collect global IDs of remote neighbours,
      // Also check neighbourFlags for correctness:
      for (size_t cell=0; cell<N_localCells; ++cell) {
	 for (size_t i=0; i<cells[cell].neighbours.size(); ++i) {
	    if (cells[cell].neighbours[i] == invalid()) {
	       if (((cells[cell].neighbourFlags >> i) & 1) != 0) {
		  std::cerr << "P#" << myrank << " LID#" << cell << " GID#" << globalIDs[cell] << " nbrFlag is one for non-existing nbr type " << i << std::endl;
	       }
	       continue;
	    }
	    if (((cells[cell].neighbourFlags >> i) & 1) != 1) {
	       std::cerr << "P#" << myrank << " LID#" << cell << " GID#" << globalIDs[cell] << " nbrFlag is zero for existing nbr type " << i;
	       std::cerr << " nbr LID#" << cells[cell].neighbours[i] << " GID#" << globalIDs[cells[cell].neighbours[i]];
	       std::cerr << std::endl;
	    }
	    
	    if (cells[cell].neighbours[i] == cell) continue;
	    ++tmpReferences[cells[cell].neighbours[i]];
	    if (cells[cell].neighbours[i] >= N_localCells) tmpRemoteCells.insert(cells[cell].neighbours[i]);
	 }
      }

      // Assume that localCells is correct. Check that remoteCells is correct.
      for (std::set<CellID>::const_iterator it=tmpRemoteCells.begin(); it!=tmpRemoteCells.end(); ++it) {
	 const CellID localID = *it;
	 if (localID < N_localCells || localID >= cells.size()) {
	    std::cerr << "P#" << myrank << " remote cell LID#" << *it << " has invalid LID!" << std::endl;
	    success = false;
	 }
      }

      // Check that remoteCells does not contain unnecessary entries:
      for (size_t i=N_localCells; i<cells.size(); ++i) {
	 std::set<CellID>::const_iterator jt=tmpRemoteCells.find(i);
	 if (jt == tmpRemoteCells.end()) {
	    std::cerr << "P#" << myrank << " unnecessary remote cell entry in cells, LID#" << i << std::endl;
	    success = false;
	 }
      }
      
      // Check that localCells and remoteCells do not contain duplicate cells:
      std::set<CellID> tmpGlobalIDs;
      for (size_t i=0; i<cells.size(); ++i) {
	 std::set<CellID>::const_iterator it = tmpGlobalIDs.find(globalIDs[i]);
	 if (tmpGlobalIDs.find(globalIDs[i]) != tmpGlobalIDs.end()) {
	    std::cerr << "P#" << myrank << " LID#" << i << " GID#" << globalIDs[i] << " duplicate entry" << std::endl;
	    success = false;
	 }
	 tmpGlobalIDs.insert(globalIDs[i]);
      }
      
      // Check that all hosts are valid:
      for (size_t i=0; i<cells.size(); ++i) {
	 if (i < N_localCells) {
	    // Check that all local cells have this process as their host:
	    if (hosts[i] != getRank()) {
	       std::cerr << "P#" << myrank << " LID#" << i << " GID#" << globalIDs[i] << " host " << hosts[i] << " should be " << getRank() << std::endl;
	       success = false;
	    }
	 } else {
	    // Remote cells must have sensible host value:
	    if (hosts[i] >= getProcesses()) {
	       std::cerr << "P#" << myrank << " LID#" << i << " GID#" << globalIDs[i] << " host " << hosts[i] << std::endl;
	       success = false;
	    }
	 }
	 if (i >= N_localCells) continue;
	 
	 // Check that all neighbour hosts have reasonable values:
	 for (size_t n=0; n<cells[i].neighbours.size(); ++n) {
	    if (cells[i].neighbours[n] == invalid()) continue;
	 }
      }
      
      // Check that all remote cells have a correct host. Each process sends everyone else
      // a list of cells it owns. Hosts can be checked based on this information:
      for (MPI_processID p = 0; p < getProcesses(); ++p) {
	 CellID N_cells;
	 if (p == getRank()) {
	    // Tell everyone how many cells this process has:
	    N_cells = N_localCells;
	    MPI_Bcast(&N_cells,1,MPI_Type<CellID>(),p,comm);
	    
	    // Create an array containing cell IDs and broadcast:
	    CellID* buffer = const_cast<CellID*>(&(globalIDs[0]));
	    MPI_Bcast(buffer,N_cells,MPI_Type<CellID>(),p,comm);
	 } else {
	    // Receive number of cells process p is sending:
	    MPI_Bcast(&N_cells,1,MPI_Type<CellID>(),p,comm);
	    // Allocate array for receiving cell IDs and receive:
	    CellID* remoteCells = new CellID[N_cells];
	    MPI_Bcast(remoteCells,N_cells,MPI_Type<CellID>(),p,comm);
	    
	    for (CellID c=0; c<N_cells; ++c) {
	       // Check that received cells are not local to this process:
	       for (size_t i=0; i<N_localCells; ++i) {
		  if (globalIDs[i] == remoteCells[c]) {
		     std::cerr << "P#" << myrank << " GID#" << remoteCells[c] << " from P#" << p << " is local to this process!" << std::endl;
		     success = false;
		  }
	       }
	       // Check that remote cell has the correct host:
	       std::map<CellID,CellID>::const_iterator it = global2LocalMap.find(remoteCells[c]);
	       if (it != global2LocalMap.end()) {
		  const CellID localID = it->second;
		  if (localID < N_localCells) {
		     std::cerr << "P#" << myrank << " GID#" << remoteCells[c] << " from P#" << p << " is local to this process!" << std::endl;
		     success = false;
		  }
		  if (hosts[localID] != p) {
		     std::cerr << "P#" << myrank << " GID#" << remoteCells[c] << " from P#" << p << " has wrong host " << hosts[localID] << std::endl;
		     success = false;
		  }
	       }
	    }
	    delete [] remoteCells; remoteCells = NULL;
	 }
      }	    
      return success;
   }

   /** Check if partitioning has changed since last time checkPartitioningStatus 
    * was called. User must define a counter elsewhere and pass it as a parameter when 
    * calling this function. ParGrid then compares the value of that counter to internal 
    * value and if they differ, grid has been repartitioned.
    * @param counter User-defined partitioning counter. Initialize this to negative value.
    * @return If true, partitioning has changed since last time this function was called.*/
   template<class C>
   bool ParGrid<C>::checkPartitioningStatus(int& counter) const {
      #ifndef NDEBUG
         if (counter > partitioningCounter) {
	    std::cerr << "(PARGRID) ERROR: User-given counter value '" << counter << "' is invalid!" << std::endl;
	 }
      #endif
      bool rvalue = false;
      if (counter < partitioningCounter) rvalue = true;
      counter = partitioningCounter;
      return rvalue;
   }
   
   /** Finalize ParGrid. After this function returns ParGrid cannot be used 
    * without re-initialisation.
    * @return If true, ParGrid finalized successfully.
    */
   template<class C>
   bool ParGrid<C>::finalize() {
      if (initialized == false) return false;
      initialized = false;
      stencils.clear();
      delete zoltan; zoltan = NULL;
      return true;
   }
   
   template<class C>
   const std::vector<CellID>& ParGrid<C>::getBoundaryCells(unsigned int stencil,int identifier) const {
      typename std::map<unsigned int,Stencil<C> >::const_iterator it = stencils.find(stencil);
      #ifndef NDEBUG
         if (it == stencils.end()) {
	    std::cerr << "(PARGRID) ERROR: Non-existing stencil " << stencil << " requested in getBoundaryCells!" << std::endl;
	    exit(1);
	 }
      #endif
      return it->second.getBoundaryCells();
   }
   
   /** Get cell's neighbours. Non-existing neighbours have their global IDs 
    * set to value ParGrid::invalid().
    * @param localID Local ID of cell whose neighbours are requested.
    * @param Reference to vector containing neihbours' global IDs. Size 
    * of vector is always 27. Vector can be indexed with ParGrid::calcNeighbourTypeID.
    */
   template<class C>
   std::vector<CellID>& ParGrid<C>::getCellNeighbourIDs(CellID localID) {
      #ifndef NDEBUG
         if (localID >= N_localCells) {
	    std::cerr << "(PARGRID) ERROR: getCellNeighbourIDs local ID#" << localID << " is too large!" << std::endl;
	    exit(1);
	 }
      #endif
      return cells[localID].neighbours;
   }
   
   template<class C>
   MPI_Comm ParGrid<C>::getComm() const {return comm;}
   
   template<class C>
   std::vector<CellID>& ParGrid<C>::getExteriorCells() {
      if (recalculateExteriorCells == true) {
	 const unsigned int ALL_EXIST = 134217728 - 1; // This value is 2^27 - 1, i.e. integer with first 27 bits flipped
	 exteriorCells.clear();
	 for (size_t i=0; i<N_localCells; ++i) {
	    if (cells[i].neighbourFlags != ALL_EXIST) exteriorCells.push_back(i);
	 }
	 recalculateExteriorCells = false;
      }
      return exteriorCells;
   }

   /** Get global IDs of cells stored on this process. The returned vector 
    * contains both local and remote cell global IDs.
    * @return Global IDs of all cells on this process.*/
   template<class C>
   const std::vector<CellID>& ParGrid<C>::getGlobalIDs() const {return globalIDs;}
   
   /** Get host process IDs of all cells (local + remote) stored on this process.
    * @return Hosts for all cells that this process has a copy of.*/
   template<class C>
   const std::vector<MPI_processID>& ParGrid<C>::getHosts() const {return hosts;}
   
   /** Query if ParGrid has initialized correctly.
    * The value returned by this function is set in ParGrid::initialize.
    * This function does not contain thread synchronization.
    * @return If true, ParGrid is ready for use.
    */
   template<class C>
   bool ParGrid<C>::getInitialized() const {return initialized;}

   template<class C>
   const std::vector<CellID>& ParGrid<C>::getInnerCells(unsigned int stencil,int identifier) const {
      typename std::map<unsigned int,Stencil<C> >::const_iterator it = stencils.find(stencil);
      if (it == stencils.end()) {
	 std::cerr << "(PARGRID) ERROR: Non-existing stencil " << stencil << " requested in getInnerCells!" << std::endl;
	 exit(1);
      }
      return it->second.getInnerCells();
   }
   
   template<class C>
   std::vector<CellID>& ParGrid<C>::getInteriorCells() {
      if (recalculateInteriorCells == true) {
	 const unsigned int ALL_EXIST = 134217728 - 1; // This value is 2^27 - 1, i.e. integer with first 27 bits flipped
	 interiorCells.clear();
	 for (size_t i=0; i<N_localCells; ++i) {
	    if (cells[i].neighbourFlags == ALL_EXIST) interiorCells.push_back(i);
	 }
	 recalculateInteriorCells = false;
      }
      return interiorCells;
   }

   /** Get the local ID of a cell with given global ID. If the specified cell was found on this 
    * process, either as a local cell or as copy of a remote cell, the returned local ID is valid.
    * @param globalID Global ID of the cell.
    * @return Local ID of the cell if the cell was found, or numeric_limits<pargrid::CellID>::max() otherwise.*/
   template<class C>
   CellID ParGrid<C>::getLocalID(CellID globalID) const {
      std::map<CellID,CellID>::const_iterator it = global2LocalMap.find(globalID);
      #ifndef NDEBUG
         if (it == global2LocalMap.end()) {
	    std::cerr << "(PARGRID) ERROR: In getLocalID, cell with global ID#" << globalID << " does not exist on P#" << myrank << std::endl;
	    exit(1);
	 }
      #endif
      if (it == global2LocalMap.end()) return std::numeric_limits<CellID>::max();
      return it->second;
   }
   
   template<class C>
   uint32_t ParGrid<C>::getNeighbourFlags(CellID localID) const {
      #ifndef NDEBUG
         if (localID >= N_localCells) {
	    std::cerr << "(PARGRID) ERROR: Local ID#" << localID << " too large in getNeighbourFlags!" << std::endl;
	 }
      #endif
      return cells[localID].neighbourFlags;
   }
   
   /** Get a list of neighbour processes. A process is considered to be a neighbour 
    * if it has one or more this process' local cells' neighbours.
    * @return List of neighbour process IDs.
    */
   template<class C>
   const std::set<MPI_processID>& ParGrid<C>::getNeighbourProcesses() const {return nbrProcesses;}
   
   /** Get the total number of cells on this process. This includes remote cells buffered on this process.
    * @return Total number of cells hosted and buffered on this process.*/
   template<class C>
   CellID ParGrid<C>::getNumberOfAllCells() const {return cells.size();}
   
   /** Get the number of cells on this process.
    * @return Number of local cells.*/
   template<class C>
   CellID ParGrid<C>::getNumberOfLocalCells() const {return N_localCells;}
   
   /** Get the number of MPI processes in the communicator used by ParGrid.
    * The value returned by this function is set in ParGrid::initialize.
    * This function does not contain thread synchronization.
    * @return Number of MPI processes in communicator comm.
    */
   template<class C>
   MPI_processID ParGrid<C>::getProcesses() const {return N_processes;}
   
   /** Get the rank of this process in the MPI communicator used by ParGrid.
    * The value returned by this function is set in ParGrid::initialize.
    * This function does not contain thread synchronization.
    * @return MPI rank of this process in communicator comm.
    */
   template<class C>
   MPI_processID ParGrid<C>::getRank() const {return myrank;}

   /** Get the remote neighbours of a given local cell.
    * @param localID Local ID of the cell.
    * @param nbrTypeIDs Searched neighours.
    * @param nbrIDs Global IDs of searched remote neighbours are written here.
    * @param hosts MPI host processes of searched remote neighbours are written here.
    */
   template<class C>
   bool ParGrid<C>::getRemoteNeighbours(CellID localID,const std::vector<unsigned char>& nbrTypeIDs,std::vector<CellID>& nbrIDs) {
      nbrIDs.clear();
      hosts.clear();
      if (localID >= N_localCells) return false;
      
      // Iterate over given neighbour type IDs and check if this cell has 
      // those neighbours, and if those neighbours are remote:
      for (size_t n=0; n<nbrTypeIDs.size(); ++n) {
	 const unsigned char nbrType = nbrTypeIDs[n];
	 if (cells[localID].neighbours[nbrType] == invalid()) continue;
	 if (hosts[cells[localID].neighbours[nbrType]] == getRank()) continue;
	 nbrIDs.push_back(cells[localID].neighbours[nbrType]);
      }	 
      return true;
   }
   
   /** Initialize ParGrid and Zoltan. Note that MPI_Init must
    * have been called prior to calling this function. If ParGrid is used in 
    * multithreaded mode, MPI must have been initialized with MPI_Init_thread.
    * @param threadID The thread ID of the thread that is calling this function. 
    * Master thread is assumed to have thread ID pargrid::MASTER_THREAD_ID. Only the 
    * master thread is allowed to initialize ParGrid at this point, other threads will 
    * simply exit with return value true. Threads other than the master thread must 
    * call ParGrid::waitInitialization() immediately after calling ParGrid::initialize.
    * @param mpiThreadingLevel Threading level of underlying MPI library, obtained 
    * from initialization of MPI via MPI_Init_thread.
    * @param comm MPI communicator that ParGrid should use.
    * @param parameters Load balancing parameters for all hierarchical levels.
    * The parameters for each hierarchical level are given in a map, whose contents are pairs
    * formed from parameter types and their string values. These maps themselves
    * are packed into a vector, whose first item (map) is used for hierarchical level
    * 0, second item for hierarchical level 1, and so forth. Zoltan is set to use 
    * hierarchical partitioning if vector size is greater than one, otherwise the 
    * load balancing method given in the first element is used.
    * @return If true, master thread initialized ParGrid correctly. All other threads 
    * will always exit with value true.
    */
   template<class C>
   bool ParGrid<C>::initialize(MPI_Comm comm,const std::vector<std::map<InputParameter,std::string> >& parameters,int threadID,int mpiThreadingLevel) {
      zoltan = NULL;
      MPI_Comm_dup(comm,&(this->comm));
      this->mpiThreadingLevel = mpiThreadingLevel;
      
      // Get the number MPI of processes, and the rank of this MPI process, in given communicator:
      MPI_Comm_size(comm,&N_processes);
      MPI_Comm_rank(comm,&myrank);
      
      // Check that parameters vector is not empty:
      if (parameters.size() == 0) {
	 std::cerr << "PARGRID ERROR: parameters vector in constructor is empty!" << std::endl;
	 return false;
      }
      
      std::map<InputParameter,std::string> zoltanParameters;
      zoltanParameters[imbalanceTolerance]    = "IMBALANCE_TOL";
      zoltanParameters[loadBalancingMethod]   = "LB_METHOD";
      zoltanParameters[processesPerPartition] = "PROCS_PER_PART";
      
      // Parse user-defined load balancing parameters into a string,string container
      // which is more convenient to use with Zoltan:
      unsigned int level = 0;
      loadBalancingParameters.clear();
      loadBalancingParameters.resize(parameters.size());
      for (std::vector<std::map<InputParameter,std::string> >::const_iterator it=parameters.begin(); it!=parameters.end(); ++it) {
	 for (std::map<InputParameter,std::string>::const_iterator jt=it->begin(); jt!=it->end(); ++jt) {
	    std::map<InputParameter,std::string>::const_iterator kt=zoltanParameters.find(jt->first);
	    if (kt != zoltanParameters.end())
	      loadBalancingParameters[level].push_back(make_pair(kt->second,jt->second));
	 }
	 ++level;
      }

      // Determine if cell weights are calculated and passed to Zoltan:
      std::string objWeightDim;
      std::map<InputParameter,std::string>::const_iterator it = parameters.front().find(cellWeightScale);
      if (it != parameters.front().end()) {
	 cellWeightsUsed = true;
	 cellWeight      = atof(it->second.c_str());
	 objWeightDim    = "1";
      } else {
	 cellWeightsUsed = false;
	 cellWeight      = 0.0;
	 objWeightDim    = "0";
      }
      
      // Determine if edge weights are calculated and passed to Zoltan:
      std::string edgeWeightDim;
      it = parameters.front().find(edgeWeightScale);
      if (it != parameters.front().end()) {
	 edgeWeightsUsed = true;
	 edgeWeight      = atof(it->second.c_str());
	 edgeWeightDim   = "1";
      } else {
	 edgeWeightsUsed = false;
	 edgeWeight      = 0.0;
	 edgeWeightDim   = "0";
      }
      
/*      if (getRank() == 0) {
	 std::cerr << "cell weights used " << cellWeightsUsed << " dim " << objWeightDim.c_str() << std::endl;
	 std::cerr << "edge weights in use " << edgeWeightsUsed << " dim " << edgeWeightDim.c_str() << std::endl;
      }*/

      if (cellWeightsUsed == false && edgeWeightsUsed == false) {
	 std::cerr << "PARGRID ERROR: You must specify cell weight scale, edge weight scale, or both in ";
	 std::cerr << "the first element in parameters vector in constructor!" << std::endl;
	 return false;
      }

      // Create a new Zoltan object and set some initial parameters:
      zoltan = new Zoltan(comm);
      zoltan->Set_Param("NUM_GID_ENTRIES","1");
      zoltan->Set_Param("NUM_LID_ENTRIES","1");
      zoltan->Set_Param("RETURN_LISTS","ALL");
      zoltan->Set_Param("OBJ_WEIGHT_DIM",objWeightDim.c_str());
      zoltan->Set_Param("EDGE_WEIGHT_DIM",edgeWeightDim.c_str());
      zoltan->Set_Param("DEBUG_LEVEL","0");
      //zoltan->Set_Param("PHG_CUT_OBJECTIVE","CONNECTIVITY");
      //zoltan->Set_Param("CHECK_HYPERGRAPH","1");
      
      // Check if hierarchical partitioning should be enabled:
      if (parameters.size() > 1) {
	 
	 std::stringstream ss;
	 int counter = 0;
	 for (size_t i=0; i<loadBalancingParameters.size(); ++i) {
	    for (std::list<std::pair<std::string,std::string> >::const_iterator it=loadBalancingParameters[i].begin();
		 it != loadBalancingParameters[i].end(); ++it) {
	       if (it->first == "PROCS_PER_PART") {
		  if (counter > 0) ss << ',';
		  ss << it->second;
		  ++counter;
	       }
	    }
	 }
	 
	 //if (getRank() == 0) std::cerr << "TOPOLOGY = '" << ss.str() << "'" << std::endl;
	 
	 zoltan->Set_Param("LB_METHOD","HIER");
	 zoltan->Set_Param("HIER_ASSIST","1");
	 //zoltan->Set_Param("HIER_CHECKS","0");
	 //zoltan->Set_Param("HIER_DEBUG_LEVEL","0");
	 zoltan->Set_Param("TOPOLOGY",ss.str());
	 
	 // These are set because of a Zoltan bug:
	 zoltan->Set_Param("EDGE_WEIGHT_DIM","0");
	 edgeWeightsUsed = false;
	 edgeWeightDim = "0";

	 if (getRank() == 0) std::cerr << "Enabling hierarchical partitioning" << std::endl;
      } else {
	 for (std::list<std::pair<std::string,std::string> >::const_iterator it=loadBalancingParameters[0].begin();
	      it != loadBalancingParameters[0].end(); ++it) {
	    if (it->first == "PROCS_PER_PART") continue;
	    zoltan->Set_Param(it->first,it->second);
	 }
      }

      // Register Zoltan callback functions:
      zoltan->Set_Num_Obj_Fn(&cb_getNumberOfLocalCells<C>,this);
      zoltan->Set_Obj_List_Fn(&cb_getLocalCellList<C>,this);
      zoltan->Set_Num_Geom_Fn(&cb_getMeshDimension<C>,this);          // Geometrical
      //zoltan->Set_Geom_Fn(&cb_getCellCoordinates,this);
      zoltan->Set_Geom_Multi_Fn(&cb_getAllCellCoordinates<C>,this);

      //zoltan->Set_Num_Edges_Fn(&cb_getNumberOfEdges<C>,this);         // Graph
      zoltan->Set_Num_Edges_Multi_Fn(&cb_getNumberOfAllEdges<C>,this);
      //zoltan->Set_Edge_List_Fn(&cb_getCellEdges<C>,this);
      zoltan->Set_Edge_List_Multi_Fn(&cb_getAllCellEdges<C>,this);

      zoltan->Set_HG_Size_CS_Fn(&cb_getNumberOfHyperedges<C>,this);                // Hypergraph
      zoltan->Set_HG_CS_Fn(&cb_getHyperedges<C>,this);
      zoltan->Set_HG_Size_Edge_Wts_Fn(cb_getNumberOfHyperedgeWeights<C>,this);
      zoltan->Set_HG_Edge_Wts_Fn(cb_getHyperedgeWeights<C>,this);

      
      //zoltan->Set_Hier_Num_Levels_Fn(cb_getNumberOfHierarchicalLevels<C>,this);   // Hierarchical
      //zoltan->Set_Hier_Part_Fn(cb_getHierarchicalPartNumber<C>,this);
      //zoltan->Set_Hier_Method_Fn(&cb_getHierarchicalParameters<C>,this);

      initialized = true;
      return initialized;
   }
   
   template<class C>
   bool ParGrid<C>::initialLoadBalance(bool balanceLoad) {

      return false;
   }
   
   /** Return invalid cell global ID. Cell IDs obtained elsewhere may be 
    * tested against this value to see if they are valid.
    * @return Invalid cell global ID.
    */
   template<class C>
   CellID ParGrid<C>::invalid() const {return std::numeric_limits<CellID>::max();}
   
   template<class C>
   void ParGrid<C>::invalidate() {
      recalculateInteriorCells = true;
      recalculateExteriorCells = true;
      for (typename std::map<unsigned int,Stencil<C> >::iterator it=stencils.begin(); it!=stencils.end(); ++it) {
	 it->second.update();
      }
   }
   
   /** Check if a cell with given global ID exists on this process.
    * In multithreaded mode this function contains a readers-writers lock.
    * @param localID Local ID of the searched cell.
    * @return If true, the cell exists on this process.
    */
   template<class C>
   bool ParGrid<C>::localCellExists(CellID localID) {
      // Acquire read access to localCells and search for the given cellID:
      if (localID >= N_localCells) return false;
      return true;
   }
   
   /** Get a pointer to user-data of the given cell.
    * @param localID Local ID of the cell that holds the requested user data.
    * @return Pointer to user data, or NULL if a cell with the given global ID does not exist.
    */
   template<class C>
   C* ParGrid<C>::operator[](const CellID& localID) {
      // Acquire read access to localCells and search for the given cellID:
      #ifndef NDEBUG
         if (localID >= cells.size()) {
	    std::cerr << "(PARGRID) ERROR: P#" << getRank() << " LID#" << localID << " too large in operator[] !" << std::endl;
	 }
      #endif
      return &(cells[localID].userData);
   }

   /** Set partitioning mode. This only has effect if GRAPH or HYPERGRAPH 
    * partitioners is used. Valid values are partition (static load balancing),
    * repartition (dynamic load balancing), and refine (fast improvement).
    * @param pm New partitioning mode. These values are passed to Zoltan. Note 
    * that one needs to call ParGrid::balanceLoad before the new mode has any effect.
    * @return If true, new mode was successfully taken into use.*/
   template<class C>
   bool ParGrid<C>::setPartitioningMode(PartitioningMode pm) {
      bool rvalue = true;
      switch (pm) {
       case partition:
	 zoltan->Set_Param("LB_APPROACH","PARTITION");
	 break;
       case repartition:
	 zoltan->Set_Param("LB_APPROACH","REPARTITION");
	 break;
       case refine:
	 zoltan->Set_Param("LB_APPROACH","REFINE");
	 break;
       default:
	 rvalue = false;
	 break;
      }
      return rvalue;
   }
   
   template<class C>
   bool ParGrid<C>::startNeighbourExchange(unsigned int stencil,int identifier) {
      if (getInitialized() == false) return false;
      if (identifier == 0) return false;
      if (stencils.find(stencil) == stencils.end()) return false;
      return stencils[stencil].startTransfer(identifier);
   }
   
   template<class C>
   bool ParGrid<C>::syncCellHosts() {
      if (getInitialized() == false) return false;
      
      // Each process sends every other process a list of cells it owns. 
      // We can then figure out remote neighbour hosts from these lists:
      CellID N_cells;
      for (MPI_processID p = 0; p < getProcesses(); ++p) {
	 if (p == getRank()) {
	    // It is this processes turn to send a list of local cells. 
	    // First tell how many cells this process has:
	    N_cells = N_localCells;
	    MPI_Bcast(&N_cells,1,MPI_Type<CellID>(),p,comm);
	    MPI_Bcast(&(globalIDs[0]),N_cells,MPI_Type<CellID>(),p,comm);
	 } else {
	    // Receive a list of global cell IDs from another process:
	    MPI_Bcast(&N_cells,1,MPI_Type<CellID>(),p,comm);

	    CellID* remoteCells = new CellID[N_cells];
	    MPI_Bcast(remoteCells,N_cells,MPI_Type<CellID>(),p,comm);

	    // Go through the received list and check if any of 
	    // this processes remote neighbours are on that list:
	    for (CellID c=0; c<N_cells; ++c) {
	       std::map<CellID,CellID>::const_iterator it = global2LocalMap.find(remoteCells[c]);
	       if (it == global2LocalMap.end()) continue;
	       const CellID localID = it->second;
	       #ifndef NDEBUG
	          // DEBUG: Check that obtained cell is not local to this process:
	          if (localID < N_localCells) {
		     std::cerr << "(PARGRID) ERROR: P#" << getRank() << " remote cell GID#" << remoteCells[c] << " is local to this process!" << std::endl;
		     exit(1);
	          }
	          // DEBUG: Check that obtained local ID does not exceed array size:
	          if (localID >= cells.size()) {
		     std::cerr << "(PARGRID) ERROR: P#" << getRank() << " remote cell GID#" << remoteCells[c] << " local ID# ";
		     std::cerr << localID << " above array bounds!" << std::endl;
		     exit(1);
		  }
	          // DEBUG: Check that globalIDs entry has correct value:
	          if (globalIDs[localID] != remoteCells[c]) {
		     std::cerr << "(PARGRID) ERROR: P#" << getRank() << " remote cell GID#" << remoteCells[c] << " has LID#" << localID;
		     std::cerr << " but globalIDs[localID] has value " << globalIDs[localID] << std::endl;
		  }
	       #endif
	       hosts[localID] = p;
	    }
	    delete [] remoteCells; remoteCells = NULL;
	 }
      }
      return true;
   }

   template<class C>
   bool ParGrid<C>::wait(unsigned int stencil,int identifier) {
      if (getInitialized() == false) return false;
      typename std::map<unsigned int,Stencil<C> >::iterator sten = stencils.find(stencil);
      if (sten == stencils.end()) return false;
      return sten->second.wait(identifier);
   }

   // **************************************************** //
   // ***** DEFINITIONS FOR ZOLTAN CALLBACK WRAPPERS ***** // 
   // **************************************************** //

   template<class C>
   void cb_getAllCellCoordinates(void* pargrid,int N_globalEntries,int N_localEntries,int N_cellIDs,ZOLTAN_ID_PTR globalID,
				 ZOLTAN_ID_PTR localID,int N_coords,double* geometryData,int* rcode) {
      ParGrid<C>* ptr = reinterpret_cast<ParGrid<C>*>(pargrid);
      ptr->getAllCellCoordinates(N_globalEntries,N_localEntries,N_cellIDs,globalID,localID,N_coords,geometryData,rcode);
   }
   
   template<class C>
   void cb_getCellCoordinates(void* pargrid,int N_globalEntries,int N_localEntries,ZOLTAN_ID_PTR globalID,
			      ZOLTAN_ID_PTR localID,CellCoordinate* geometryData,int* rcode) {
      ParGrid<C>* ptr = reinterpret_cast<ParGrid<C>*>(pargrid);
      ptr->getCellCoordinates(N_globalEntries,N_localEntries,globalID,localID,geometryData,rcode);
   }
   
   template<class C>
   void cb_getAllCellEdges(void* pargrid,int N_globalIDs,int N_localIDs,int N_cells,ZOLTAN_ID_PTR globalIDs,
			   ZOLTAN_ID_PTR localIDs,int* N_edges,ZOLTAN_ID_PTR nbrGlobalIDs,int* nbrHosts,
			   int N_weights,CellWeight* edgeWeights,int* rcode) {
      ParGrid<C>* ptr = reinterpret_cast<ParGrid<C>*>(pargrid);
      ptr->getAllCellEdges(N_globalIDs,N_localIDs,N_cells,globalIDs,localIDs,N_edges,
			   nbrGlobalIDs,nbrHosts,N_weights,edgeWeights,rcode);
   }
   
   template<class C>
   void cb_getCellEdges(void* pargrid,int N_globalIDs,int N_localIDs,ZOLTAN_ID_PTR globalID,
			ZOLTAN_ID_PTR localID,ZOLTAN_ID_PTR nbrGlobalIDs,int* nbrHosts,
			int N_weights,CellWeight* weight,int* rcode) {
      ParGrid<C>* ptr = reinterpret_cast<ParGrid<C>*>(pargrid);
      ptr->getCellEdges(N_globalIDs,N_localIDs,globalID,localID,nbrGlobalIDs,nbrHosts,N_weights,weight,rcode);
   }
   
   template<class C>
   void cb_getHierarchicalParameters(void* pargrid,int level,Zoltan_Struct* zs,int* rcode) {
      ParGrid<C>* ptr = reinterpret_cast<ParGrid<C>*>(pargrid);
      ptr->getHierarchicalParameters(level,zs,rcode);
   }
   
   template<class C>
   int cb_getHierarchicalPartNumber(void* pargrid,int level,int* rcode) {
      ParGrid<C>* ptr = reinterpret_cast<ParGrid<C>*>(pargrid);
      return ptr->getHierarchicalPartNumber(level,rcode);
   }
   
   template<class C>
   void cb_getHyperedges(void* pargrid,int N_globalIDs,int N_vtxedges,int N_pins,int format,ZOLTAN_ID_PTR vtxedge_GID,
			 int* vtxedge_ptr,ZOLTAN_ID_PTR pin_GID,int* rcode) {
      ParGrid<C>* ptr = reinterpret_cast<ParGrid<C>*>(pargrid);
      ptr->getHyperedges(N_globalIDs,N_vtxedges,N_pins,format,vtxedge_GID,vtxedge_ptr,pin_GID,rcode);
   }
   
   template<class C>
   void cb_getHyperedgeWeights(void* pargrid,int N_globalIDs,int N_localIDs,int N_edges,int N_weights,
			       ZOLTAN_ID_PTR edgeGlobalID,ZOLTAN_ID_PTR edgeLocalID,CellWeight* edgeWeights,int* rcode) {
      ParGrid<C>* ptr = reinterpret_cast<ParGrid<C>*>(pargrid);
      ptr->getHyperedgeWeights(N_globalIDs,N_localIDs,N_edges,N_weights,edgeGlobalID,edgeLocalID,edgeWeights,rcode);
   }
   
   template<class C>
   void cb_getLocalCellList(void* pargrid,int N_globalIDs,int N_localIDs,ZOLTAN_ID_PTR globalIDs,
			    ZOLTAN_ID_PTR localIDs,int N_weights,CellWeight* cellWeights,int* rcode) {
      ParGrid<C>* ptr = reinterpret_cast<ParGrid<C>*>(pargrid);
      ptr->getLocalCellList(N_globalIDs,N_localIDs,globalIDs,localIDs,N_weights,cellWeights,rcode);
   }
   
   template<class C>
   int cb_getMeshDimension(void* pargrid,int* rcode) {
      ParGrid<C>* ptr = reinterpret_cast<ParGrid<C>*>(pargrid);
      return ptr->getMeshDimension(rcode);
   }
   
   template<class C>
   void cb_getNumberOfAllEdges(void* pargrid,int N_globalIDs,int N_localIDs,int N_cells,ZOLTAN_ID_PTR globalIDs,
			       ZOLTAN_ID_PTR localIDs,int* N_edges,int* rcode) {
      ParGrid<C>* ptr = reinterpret_cast<ParGrid<C>*>(pargrid);
      ptr->getNumberOfAllEdges(N_globalIDs,N_localIDs,N_cells,globalIDs,localIDs,N_edges,rcode);
   }
   
   template<class C>
   int cb_getNumberOfEdges(void* pargrid,int N_globalIDs,int N_localIDs,ZOLTAN_ID_PTR globalID,ZOLTAN_ID_PTR localID,int* rcode) {
      ParGrid<C>* ptr = reinterpret_cast<ParGrid<C>*>(pargrid);
      return ptr->getNumberOfEdges(N_globalIDs,N_localIDs,globalID,localID,rcode);
   }
   
   template<class C>
   int cb_getNumberOfHierarchicalLevels(void* pargrid,int* rcode) {
      ParGrid<C>* ptr = reinterpret_cast<ParGrid<C>*>(pargrid);
      return ptr->getNumberOfHierarchicalLevels(rcode);
   }
   
   template<class C>
   void cb_getNumberOfHyperedges(void* pargrid,int* N_lists,int* N_pins,int* format,int* rcode) {
      ParGrid<C>* ptr = reinterpret_cast<ParGrid<C>*>(pargrid);
      ptr->getNumberOfHyperedges(N_lists,N_pins,format,rcode);
   }
   
   template<class C>
   void cb_getNumberOfHyperedgeWeights(void* pargrid,int* N_edges,int* rcode) {
      ParGrid<C>* ptr = reinterpret_cast<ParGrid<C>*>(pargrid);
      ptr->getNumberOfHyperedgeWeights(N_edges,rcode);
   }
   
   template<class C>
   int cb_getNumberOfLocalCells(void* pargrid,int* rcode) {
      ParGrid<C>* ptr = reinterpret_cast<ParGrid<C>*>(pargrid);
      return ptr->getNumberOfLocalCells(rcode);
   }

   // ***************************************************** //
   // ***** DEFINITIONS FOR ZOLTAN CALLBACK FUNCTIONS ***** //
   // ***************************************************** //
   
   /** Definition of Zoltan callback function ZOLTAN_GEOM_MULTI_FN. This function is required by geometry-based 
    * load balancing (BLOCK,RCB,RIB,HSFC,REFTREE) functions. The purpose of this function is to tell Zoltan 
    * the physical x/y/z coordinates of all cells local to this process.
    * @param N_globalIDs Size of global ID.
    * @param N_localIDs Size of local ID.
    * @param N_cellIDs Number of cells whose coordinates are requested.
    * @param globalIDs Global IDs of cells whose coordinates are requested.
    * @param localIDs Local IDs of cells whose coordinates are requested.
    * @param N_coords Dimensionality of the mesh.
    * @param geometryData Array in which cell coordinates are to be written.
    * @param rcode Return code, upon successful exit value ZOLTAN_OK is written here.
    */
   template<class C>
   void ParGrid<C>::getAllCellCoordinates(int N_globalIDs,int N_localIDs,int N_cellIDs,ZOLTAN_ID_PTR globalIDs,
					  ZOLTAN_ID_PTR localIDs,int N_coords,double* geometryData,int* rcode) {
      typename std::map<CellID,ParCell<C> >::const_iterator it;
      for (int i=0; i<N_cellIDs; ++i) {
	 cells[localIDs[i]].userData.getCoordinates(geometryData + i*N_coords);
      }
      *rcode = ZOLTAN_OK;
   }
   
   /** Definition of Zoltan callback function ZOLTAN_GEOM_FN. This function is required by
    * geometry-based load balancing (BLOCK,RCB,RIB,HSFC,Reftree). The purpose of this function is to tell
    * Zoltan the physical x/y/z coordinates of a given cell on this process.
    * In multithreaded mode this function blocks until a read lock is acquired for ParGrid::localCells.
    * @param N_globalEntries The size of array globalID.
    * @param N_localEntries The size of array localID.
    * @param globalID The global ID of the cell whose coordinates are queried.
    * @param localID The local ID of the cell whose coordinates are queried.
    * @param geometryData Array where the coordinates should be written. Zoltan has reserved this array, its size
    * is determined by the getGridDimensions function.
    * @param rcode The return code. Upon success this should be ZOLTAN_OK.
    */
   template<class C>
   void ParGrid<C>::getCellCoordinates(int N_globalEntries,int N_localEntries,ZOLTAN_ID_PTR globalID,
				       ZOLTAN_ID_PTR localID,CellCoordinate* geometryData,int* rcode) {
      // Check that the given cell exists on this process:
      #ifdef DEBUG_PARGRID
         if (localID >= N_localCells) {
	    std::cerr << "ParGrid ERROR: getCellCoordinates queried non-existing cell #" << localID << std::endl;
	    *rcode = ZOLTAN_FATAL;
	    return;
	 }
      #endif

      // Get cell coordinates from user:
      cells[localID].userData.getCoordinates(geometryData);
      *rcode = ZOLTAN_OK;
   }

   /** Definition for Zoltan callback function ZOLTAN_EDGE_LIST_MULTI_FN. This function is required
    * for graph-based load balancing (GRAPH). The purpose of this function is to tell Zoltan
    * the global IDs of each neighbour of all cells local to this process, as well as the ranks of the 
    * MPI processes who own the neighbours and the weights of the edges (if edge weights are used).
    * @param N_globalIDs Size of global ID.
    * @param N_localIDs Size of local ID.
    * @param N_cells
    * @param globalIDs Global IDs cells whose neighbours are queried.
    * @param localIDs Local IDs cells whose neighbours are queried.
    * @param nbrGlobalIDs Array in which the global IDs of neighbours of all given cells are to be written.
    * @param nbrHosts For each neighbour, the rank of the MPI process who owns the cell.
    * @param N_weights The size of edge weight.
    * @param weight Array in which the weights of each edge are to be written.
    * @param rcode Return code, upon successful exit value ZOLTAN_OK is written here.
    */
   template<class C>
   void ParGrid<C>::getAllCellEdges(int N_globalIDs,int N_localIDs,int N_cells,ZOLTAN_ID_PTR globalIDs,
				    ZOLTAN_ID_PTR localIDs,int* N_edges,ZOLTAN_ID_PTR nbrGlobalIDs,int* nbrHosts,
				    int N_weights,CellWeight* edgeWeights,int* rcode) {
      size_t counter = 0;
      if (N_weights == 0) {
	 // Edge weights are not calculated
	 for (int i=0; i<N_cells; ++i) {
	    // Copy cell's neighbour information:
	    for (size_t j=0; j<cells[localIDs[i]].neighbours.size(); ++j) {
	       const CellID nbrLocalID = cells[localIDs[i]].neighbours[j];
	       if (nbrLocalID == invalid()) continue;
	       
	       // Copy neighbour global ID and host process ID:
	       nbrGlobalIDs[counter] = this->globalIDs[nbrLocalID];
	       nbrHosts[counter]     = hosts[nbrLocalID];
	       ++counter;
	    }
	 }
      } else {
	 // Edge weights are calculated
	 for (int i=0; i<N_cells; ++i) {
	    // Copy cell's neighbour information:
	    for (size_t j=0; j<cells[localIDs[i]].neighbours.size(); ++j) {
	       const CellID nbrLocalID = cells[localIDs[i]].neighbours[j];
	       if (nbrLocalID == invalid()) continue;

	       // Copy neighbour global ID and host process ID:
	       nbrGlobalIDs[counter] = this->globalIDs[nbrLocalID];
	       nbrHosts[counter]     = hosts[nbrLocalID];
	       edgeWeights[counter]  = edgeWeight*cells[nbrLocalID].userData.getWeight();
	       ++counter;
	    }
	 }	    
      }
      *rcode = ZOLTAN_OK;
   }
   
   /** Definition for Zoltan callback function ZOLTAN_EDGE_LIST_FN. This function is required
    * for graph-based load balancing (GRAPH). The purpose is to give the global IDs of each neighbour of
    * a given cell, as well as the ranks of the MPI processes which have the neighbouring cells.
    * In multithreaded mode this function blocks until a read lock is acquired on localCells.
    * @param N_globalIDs The size of array globalID.
    * @param N_localIDs The size of array localID.
    * @param globalID The global ID of the cell whose neighbours are queried.
    * @param localID The local ID of the cell whose neighbours are queried.
    * @param nbrGlobalIDs An array where the global IDs of the neighbours are written. Note that
    * Zoltan has already allocated this array based on a call to getNumberOfEdges function.
    * @param nbrHosts For each neighbour, the rank of the MPI process which has the cell.
    * @param N_weights The size of array weight.
    * @param weight The weight of each edge.
    * @param rcode The return code. Upon success should be ZOLTAN_OK.
    */
   template<class C>
   void ParGrid<C>::getCellEdges(int N_globalIDs,int N_localIDs,ZOLTAN_ID_PTR globalID,
				   ZOLTAN_ID_PTR localID,ZOLTAN_ID_PTR nbrGlobalIDs,int* nbrHosts,
				   int N_weights,CellWeight* weight,int* rcode) {
      // Count global IDs of cell's neighbours into Zoltan structures and calculate 
      // edge weight (if in use):
      int counter = 0;
      if (edgeWeightsUsed == true) {
	 for (size_t i=0; i<cells[localID[0]].neighbours.size(); ++i) {
	    if (cells[localID[0]].neighbours[i] == invalid()) continue;
	    const CellID nbrLocalID = cells[localID[0]].neighbours[i];

	    nbrGlobalIDs[counter] = globalIDs[nbrLocalID];
	    nbrHosts[counter]     = hosts[nbrLocalID];
	    weight[counter]       = edgeWeight;
	    ++counter;
	 }
      } else {
	 for (size_t i=0; i<cells[localID[0]].neighbours.size(); ++i) {
	    if (cells[localID[0]].neighbours[i] == invalid()) continue;
	    const CellID nbrLocalID = cells[localID[0]].neighbours[i];
	    nbrGlobalIDs[counter] = globalIDs[nbrLocalID];
	    nbrHosts[counter]     = cells[nbrLocalID];
	    ++counter;
	 }
      }
      *rcode = ZOLTAN_OK;
   }
   
   template<class C>
   void ParGrid<C>::getHierarchicalParameters(int level,Zoltan_Struct* zs,int* rcode) {
      #ifndef DEBUG
         // Sanity check on input parameters:
         if (level < 0 || level >= (int)loadBalancingParameters.size()) {*rcode = ZOLTAN_FATAL; return;}
      #endif
      
      // Copy user-defined load balancing parameters to Zoltan structure:
      for (std::list<std::pair<std::string,std::string> >::const_iterator it=loadBalancingParameters[level].begin();
	   it!=loadBalancingParameters[level].end(); ++it) {
	 if (it->first == "PROCS_PER_PART") continue;
	 Zoltan_Set_Param(zs,it->first.c_str(),it->second.c_str());
      }
   }
   
   template<class C>
   int ParGrid<C>::getHierarchicalPartNumber(int level,int* rcode) {
      #ifndef NDEBUG
         if (level < 0 || level >= loadBalancingParameters.size()) {*rcode = ZOLTAN_FATAL; return -1;}
      #endif
      
      MPI_processID rank = getRank();
      for (int i=0; i<level; ++i) {
	 bool found = false;
	 int procsPerPart = 0;
	 for (std::list<std::pair<std::string,std::string> >::const_iterator it=loadBalancingParameters[i].begin();
	      it != loadBalancingParameters[i].end(); ++it) {
	    if (it->first == "PROCS_PER_PART") {
	       procsPerPart = atoi(it->second.c_str());
	       found = true;
	    }
	 }
	 
	 if (found == false) {
	    *rcode = ZOLTAN_FATAL;
	    return -1;
	 }
	 
	 if (i == level) {
	    rank = rank / procsPerPart;
	 } else {
	    rank = rank % procsPerPart;
	 }
      }
      
      *rcode = ZOLTAN_OK;
      return rank;
   }
   
   /** Definition for Zoltan callback function ZOLTAN_HG_CS_FN. This function is required for
    * hypergraph-based load balancing (HYPERGRAPH). The purpose is to give Zoltan the hypergraph in a compressed format.
    * In multithreaded mode this function blocks until read lock is acquired on ParGrid::localCells.
    * @param N_globalIDs The size of globalID.
    * @param N_vtxedges The number of entries that need to be written to vtxedge_GID.
    * @param N_pins The number of pins that need to be written to pin_GID.
    * @param format The format that is used to represent the hypergraph, either ZOLTAN_COMPRESSED_EDGE or ZOLTAN_COMPRESSED_VERTEX.
    * @param vtxedge_GID An array where the hypergraph global IDs are written into.
    * @param vtxedge_ptr An array where, for each hyperedge, an index into pin_GID is given from where the pins for that
    * hyperedge are given.
    * @param pin_GID An array where the pins are written to.
    * @param rcode The return code. Upon success should be ZOLTAN_OK.
    */
   template<class C>
   void ParGrid<C>::getHyperedges(int N_globalIDs,int N_vtxedges,int N_pins,int format,ZOLTAN_ID_PTR vtxedge_GID,
				  int* vtxedge_ptr,ZOLTAN_ID_PTR pin_GID,int* rcode) {
      // Check that correct hyperedge format is requested:
      if (format != ZOLTAN_COMPRESSED_VERTEX) {
	 *rcode = ZOLTAN_FATAL;
	 return;
      }

      int pinCounter = 0;

      // Create list of hyperedges and pins:
      for (CellID i=0; i<N_localCells; ++i) {
	 vtxedge_GID[i]      = globalIDs[i];
	 vtxedge_ptr[i]      = pinCounter;
	 pin_GID[pinCounter] = globalIDs[i];
	 
	 // Add pin to this cell and to every existing neighbour:
	 for (size_t j=0; j<cells[i].neighbours.size(); ++j) {
	    if (cells[i].neighbours[j] == invalid()) continue;
	    pin_GID[pinCounter] = globalIDs[cells[i].neighbours[j]];
	    ++pinCounter;
	 }
      }
      *rcode = ZOLTAN_OK;
   }

   /** Definition for Zoltan callback function ZOLTAN_HG_EDGE_WTS_FN. This is an optional function
    * for hypergraph-based load balancing (HYPEREDGE). The purpose is to tell Zoltan the weight of each hyperedge.
    * @param N_globalIDs The size of edgeGlobalID entry.
    * @param N_localIDs The size of edgeLocalID entry.
    * @param N_edges The number of hyperedge weights that need to be written to edgeWeights.
    * @param N_weights Number of weights per hyperedge that need to be written.
    * @param edgeGlobalID An array where the global IDs of each weight-supplying hyperedge are to be written.
    * @param edgeLocalID An array where the local IDs of each weight-supplying hyperedge are to be written.
    * This array can be left empty.
    * @param edgeWeights An array where the hyperedge weights are written into.
    * @param rcode The return code. Upon success should be ZOLTAN_OK.
    */
   template<class C>
   void ParGrid<C>::getHyperedgeWeights(int N_globalIDs,int N_localIDs,int N_edges,int N_weights,
					ZOLTAN_ID_PTR edgeGlobalID,ZOLTAN_ID_PTR edgeLocalID,CellWeight* edgeWeights,int* rcode) {
      unsigned int counter = 0;
      
      if (edgeWeightsUsed == true) {
	 for (CellID i=0; i<N_localCells; ++i) {
	    edgeGlobalID[counter] = globalIDs[i];
	    edgeWeights[counter]  = edgeWeight * cells[i].userData.getWeight();
	    ++counter;
	 }
      } else {
	 for (CellID i=0; i<N_localCells; ++i) {
	    edgeGlobalID[counter] = globalIDs[i];
	    ++counter;
	 }
      }
      *rcode = ZOLTAN_OK;
   }

   /** Definition for Zoltan callback function ZOLTAN_OBJ_LIST_FN. This function
    * is required to use Zoltan. The purpose is to tell Zoltan the global and local
    * IDs of the cells assigned to this process, as well as their weights.
    * @param N_globalIDs The number of array entries used to describe one global ID.
    * @param N_localIDs The number of array entries used to describe one local ID.
    * @param globalIDs An array which is to be filled with the global IDs of the cells
    * currently assigned to this process. This array has been allocated by Zoltan.
    * @param localIDs An array which is to be filled with the local IDs of the cells
    * currently assigned to this process. This array has been allocated by Zoltan.
    * @param N_weights
    * @param cellWeights An array which is to be filled with the cell weights.
    * @param rcode The return code. Upon success this should be ZOLTAN_OK.
    */
   template<class C>
   void ParGrid<C>::getLocalCellList(int N_globalIDs,int N_localIDs,ZOLTAN_ID_PTR globalIDs,
				     ZOLTAN_ID_PTR localIDs,int N_weights,CellWeight* cellWeights,int* rcode) {
      #ifndef NDEBUG
         if (N_globalIDs != 1 || N_localIDs != 1) {
	    std::cerr << "(PARGRID) ERROR: Incorrect number of global/local IDs!" << std::endl;
	    exit(1);
	 }
      #endif
      
      CellID counter = 0;
      if (N_weights == 1) {
	 // Iterate over all local cells, and get the cell weights from user. This 
	 // allows support for variable cell weights.
	 for (CellID i=0; i<N_localCells; ++i) {
	    globalIDs[counter]   = this->globalIDs[i];
	    localIDs[counter]    = i;
	    cellWeights[counter] = cells[i].userData.getWeight();
	    ++counter;
	 }
      } else {
	 // Iterate over all local cells and just copy global IDs to Zoltan structures:
	 for (CellID i=0; i<N_localCells; ++i) {
	    globalIDs[counter] = this->globalIDs[i];
	    localIDs[counter]  = i;
	    ++counter;
	 }
      }
      *rcode = ZOLTAN_OK;
   }
 
   /** Definition for Zoltan callback function ZOLTAN_NUM_GEOM_FN. This function is
    * required for geometry-based load balancing (BLOCK,RCB,RIB,HSFC,Reftree).
    * The purpose is 
    * to tell Zoltan the dimensionality of the grid. ParGrid always uses 
    * three-dimensional mesh internally. In multithreaded mode this function 
    * does not contain thread synchronization.
    * @param parGridPtr A pointer to ParGrid.
    * @param rcode The return code. Upon success this should be ZOLTAN_OK.
    * @return The number of physical dimensions. Returns a value three.
    */
   template<class C>
   int ParGrid<C>::getMeshDimension(int* rcode) {
      *rcode = ZOLTAN_OK;
      return 3;
   }

   /** Definition of Zoltan callback function ZOLTAN_NUM_EDGES_MULTI_FN. This function is required 
    * for graph-based load balancing (GRAPH). The purpose of this function is to tell Zoltan how 
    * many edges each cell local to this process has.
    * @param N_globalIDs Size of global ID.
    * @param N_localIDs Size of local ID.
    * @param N_cells Number of cells whose number of edges are requested.
    * @param globalIDs Global IDs of cells whose number of edges are requested.
    * @param localIDs Local IDs of cells whose number of edges are requested.
    * @param N_edges Array in which the number of edges each cell has are to be written.
    * @param rcode Return code, upon successful exit a value ZOLTAN_OK is written here.
    */
   template<class C>
   void ParGrid<C>::getNumberOfAllEdges(int N_globalIDs,int N_localIDs,int N_cells,ZOLTAN_ID_PTR globalIDs,
				       ZOLTAN_ID_PTR localIDs,int* N_edges,int* rcode) {
      for (int i=0; i<N_cells; ++i) {
	 const CellID localID = localIDs[i];
	 int edgeSum = 0;
	 for (size_t n=0; n<cells[localID].neighbours.size(); ++n) {
	    if (cells[localID].neighbours[n] == invalid()) continue;
	    ++edgeSum;
	 }
	 N_edges[i] = edgeSum;
      }
      *rcode = ZOLTAN_OK;
   }
   
   /** Definition of Zoltan callback function ZOLTAN_NUM_EDGES_FN. This function is required
    * for graph-based load balancing (GRAPH). The purpose is to tell how many edges a given cell has, i.e.
    * how many neighbours it has to share data with.
    * In multithreaded mode this function blocks until a read lock is acquired for ParGrid::localCells.
    * @param N_globalIDs The size of array globalID.
    * @param N_localIDs The size of array localID.
    * @param globalID The global ID of the cell whose edges are queried.
    * @param localID The local ID of the cell whose edges are queried.
    * @param rcode The return code. Upon success this should be ZOLTAN_OK.
    * @return The number of edges the cell has. For three-dimensional box grid this is between 3 and 6,
    * depending on if the cell is on the edge of the simulation volume.
    */
   template<class C>
   int ParGrid<C>::getNumberOfEdges(int N_globalIDs,int N_localIDs,ZOLTAN_ID_PTR globalID,
				    ZOLTAN_ID_PTR localID,int* rcode) {
      // Count the number of neighbours the cell has:
      const CellID LID = localID[0];
      int edgeSum = 0;
      for (size_t n=0; n<cells[LID].neighbours.size(); ++n) {
	 if (cells[LID].neighbours[n] == invalid()) continue;
	 ++edgeSum;
      }
      
      // Return the number of edges:
      *rcode = ZOLTAN_OK;
      return edgeSum;
   }
   
   template<class C>
   int ParGrid<C>::getNumberOfHierarchicalLevels(int* rcode) {
      *rcode = ZOLTAN_OK;
      return loadBalancingParameters.size();
   }

   /** Definition for Zoltan callback function ZOLTAN_HG_SIZE_CS_FN. This function is required
    * for hypergraph-based load balancing (HYPERGRAPH). The purpose is to tell Zoltan which hypergraph format
    * is used (ZOLTAN_COMPRESSED_EDGE or ZOLTAN_COMPRESSED_VERTEX), how many hyperedges and
    * vertices there will be, and how many pins.
    * In multithreaded mode this function blocks until a read lock is acquired on ParGrid::localCells.
    * @param N_lists The total number of vertices or hyperedges (depending on the format)
    * is written to this variable.
    * @param N_pins The total number of pins (connections between vertices and hyperedges) is 
    * written to this variable.
    * @param format The chosen hyperedge storage format is written to this variable.
    * @param rcode The return code. Upon success should be ZOLTAN_OK.
    */
   template<class C>
   void ParGrid<C>::getNumberOfHyperedges(int* N_lists,int* N_pins,int* format,int* rcode) {
      *format = ZOLTAN_COMPRESSED_VERTEX;
      
      // Each local cell is a vertex:
      *N_lists = N_localCells;

      // Calculate the total number of pins:
      int totalNumberOfPins = 0;
      for (CellID i=0; i<N_localCells; ++i) {
	 for (size_t n=0; n<cells[i].neighbours.size(); ++n) {
	    if (cells[i].neighbours[n] == invalid()) continue;
	    ++totalNumberOfPins;
	 }
      }
      *N_pins = totalNumberOfPins;
      *rcode = ZOLTAN_OK;
   }
   
   /** Definition for Zoltan callback function ZOLTAN_HG_SIZE_EDGE_WTS_FN. This is an optional function
    * for hypergraph-based load balancing (HYPEREDGE). The purpose is to tell Zoltan how many hyperedges will have
    * a weight factor. Here we give a weight to each hyperedge.
    * @param parGridPtr A pointer to ParGrid.
    * @param N_edges A parameter where the number of weight-supplying hyperedges is written into.
    * @param rcode The return code. Upon success should be ZOLTAN_OK.
    */
   template<class C>
   void ParGrid<C>::getNumberOfHyperedgeWeights(int* N_edges,int* rcode) {
      *N_edges = N_localCells;
      *rcode = ZOLTAN_OK;
   }
   
   /** Definition for Zoltan callback function ZOLTAN_NUM_OBJ_FN. This function
    * is required to use Zoltan. The purpose is to tell Zoltan how many cells
    * are currently assigned to this process. In multithreaded mode this 
    * function blocks until a reader lock is acquired on ParGrid::localCells.
    * @param rcode The return code. Upon success this should be ZOLTAN_OK.
    * @return The number of cells assigned to this process.
    */
   template<class C>
   int ParGrid<C>::getNumberOfLocalCells(int* rcode) {
      // Get the size of localCells container:
      *rcode = ZOLTAN_OK;
      return N_localCells;
   }
   
}

#endif
