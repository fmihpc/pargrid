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

// Includes for external package headers:

#include <mpi.h>
#include <zoltan_cpp.h>

#include "mpiconversion.h"

#include <ctime>

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
      loadBalancingMethod                           /**< Load balancing method to use.*/
   };

   enum StencilType {
      localToRemoteUpdates,
      remoteToLocalUpdates
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
      std::vector<CellID> neighbours;              /**< Array of size N_neighbours, neighbour global IDs.*/
      std::vector<unsigned char> neighbourTypes;   /**< Array of size N_neighbours, neighbour type IDs.*/
      std::vector<unsigned char> refNumbers;       /**< Array of size refinementLevel, cell refinement numbers.*/
      std::vector<MPI_processID> neighbourHosts;
      MPI_processID host;                          /**< MPI rank of the process that owns this cell. For local 
						    * cells this value is the same as the rank of this process, 
						    * for boundary (buffer) cells it is the rank of the process 
						    * who owns this cell.*/
      uint32_t neighbourFlags;                     /**< Neighbour existence flags. Each cell existing within a 3x3x3 
						    * cube of cells, in which this cells sits at the center, has its 
						    * corresponding bit in neighbourFlags flipped to unit value.
						    * This variable has undefined values for remote cells.
						    */
      C userData;                                  /**< User data.*/

      void getData(bool sending,int ID,int* blockLengths,MPI_Aint* displacements,MPI_Datatype* types);
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
      bool startTransfer(int ID);
      bool update();
      bool wait(int ID);

    private:
      bool calcLocalUpdateSendsAndReceives();
      bool calcRemoteUpdateSendsAndReceives();
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
      
      std::vector<CellID> boundaryCells;
      std::vector<CellID> innerCells;
      bool initialized;                                            /**< If true, Stencil has initialized successfully and is ready for use.*/
      std::vector<unsigned char> receivedNbrTypeIDs;               /**< Neighbour type IDs indicating which cells to receive data from.*/
      std::vector<unsigned char> sentNbrTypeIDs;                   /**< Neighbour type IDs indicating which cells to send data.*/
      std::map<int,std::map<MPI_processID,TypeCache> > typeCaches; /**< MPI datatype caches for each transfer identifier,
								    * one cache per neighbouring process.*/
      std::map<int,TypeInfo> typeInfo;                             /**< Additional data transfer information for each 
								    * transfer identifier.*/
      ParGrid<C>* parGrid;                                         /**< Pointer to parallel grid.*/
      StencilType stencilType;
      std::map<MPI_processID,std::set<CellID> > recvs;
      std::map<MPI_processID,std::set<CellID> > sends;
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
      bool addTransfer(unsigned int stencil,int identifier,bool recalculate);
      bool balanceLoad();
      void barrier(int threadID=0) const;
      void calcNeighbourOffsets(unsigned char nbrTypeID,int& i_off,int& j_off,int& k_off) const;
      unsigned char calcNeighbourTypeID(int i_off,int j_off,int k_off) const;
      bool finalize();
      const std::vector<CellID>& getBoundaryCells(unsigned int stencil,int identifier) const;
      std::vector<MPI_processID>& getCellNeighbourHosts(CellID cellID);
      std::vector<CellID>& getCellNeighbourIDs(CellID cellID);
      MPI_Comm getComm() const;
      std::vector<CellID>& getExteriorCells();
      bool getInitialized() const;
      std::vector<CellID>& getInteriorCells();
      void getLocalCellIDs(std::vector<CellID>& cells) const;
      const std::set<MPI_processID>& getNeighbourProcesses() const;
      int getNeighbourReferenceCount(CellID cellID);
      CellID getNumberOfLocalCells() const;
      MPI_processID getProcesses() const;
      MPI_processID getRank() const;
      void getRemoteCellIDs(std::vector<CellID>& cells) const;
      bool getRemoteNeighbours(CellID cellID,const std::vector<unsigned char>& nbrTypeIDs,std::vector<CellID>& nbrIDs,std::vector<MPI_processID>& hosts);
      bool initialize(MPI_Comm comm,const std::vector<std::map<InputParameter,std::string> >& parameters,int threadID=0,int mpiThreadingLevel=0);
      bool initialLoadBalance(bool balanceLoad=true);
      CellID invalid() const;
      bool localCellExists(CellID cellID);
      C* operator[](const CellID& cellID);
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
//      static void getHierarchicalParameters(void* parGridPtr,int level,Zoltan_Struct* zs,int* rcode);
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
									   * missing neighbours. This list is updated only when needed.*/
      std::vector<CellID> interiorCells;                                  /**< List of interior cells on this process, i.e. cells with 
									   * zero missing neighbours. This list is recalculated only when needed.*/
      
      CellWeight cellWeight;                                              /**< Cell weight scale, used to calculate cell weights for Zoltan.*/
      bool cellWeightsUsed;                                               /**< If true, cell weights are calculated.*/
      MPI_Comm comm;                                                      /**< MPI communicator used by ParGrid.*/
      typename std::map<CellID,ParCell<C> >::iterator currentCell;        /**< Iterator to one of the local cells.*/
      CellID currentCellID;                                               /**< Global ID of cell which is currently loaded to currentCell.*/
      CellWeight edgeWeight;                                              /**< Edge weight scale, used to calculate edge weights for Zoltan.*/
      bool edgeWeightsUsed;                                               /**< If true, edge weights are calculated.*/
      bool initialized;                                                   /**< If true, ParGrid initialized correctly and is ready for use.*/
      std::vector<std::list<std::pair<std::string,std::string> > > 
	                                         loadBalancingParameters; /**< LB parameters for Zoltan for each hierarchical level.*/
      std::map<CellID,ParCell<C> > localCells;
      int mpiThreadingLevel;                                              /**< Threading level in which MPI was initialized.*/
      MPI_processID myrank;                                               /**< MPI rank of this process in communicator comm.*/
      std::set<MPI_processID> nbrProcesses;                               /**< MPI ranks of neighbour processes. A process is considered 
									   * to be a neighbour if it has any of this processes local 
									   * cells' neighbours. Calculated in balanceLoad.*/
      std::map<CellID,int> nbrReferences;
      MPI_processID N_processes;                                          /**< Number of MPI processes in communicator comm.*/
      std::vector<MPI_Request> recvRequests;
      std::map<CellID,ParCell<C> > remoteCells;                           /**< Remote neighbours of this processes local cells.*/
      std::vector<MPI_Request> sendRequests;
      std::map<unsigned int,Stencil<C> > stencils;
      Zoltan* zoltan;                                                     /**< Pointer to Zoltan.*/

      std::map<CellID,int>::iterator addAndIncreaseNeighbourReferenceEntry(CellID nbrID);
      bool checkInternalStructures() const;
      void invalidate();
      void reduceReferenceCount(CellID nbrID);
      bool syncCellHosts();
   };

   // *************************************************
   // ***** PARCELL TEMPLATE FUNCTION DEFINITIONS *****
   // *************************************************
   
   template<class C>
   ParCell<C>::ParCell(): host(std::numeric_limits<MPI_processID>::max()),neighbourFlags(0) {
      info[0] = 0;
      info[1] = 1;
      refNumbers.resize(1);
      neighbourHosts.resize(27);
   }
   
   template<class C>
   ParCell<C>::~ParCell() { }

   template<class C>
   void ParCell<C>::getData(bool sending,int ID,int* blockLengths,MPI_Aint* displacements,MPI_Datatype* types) {
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
	 userData.getData(sending,ID,blockLengths+3,displacements+3,types+3);
      } else {
	 // Cells are not migrated, only some cell data is transferred. 
	 // Get data from user:
	 userData.getData(sending,ID,blockLengths,displacements,types);
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
   void Stencil<C>::clear() {
      boundaryCells.clear();
      innerCells.clear();
      recvs.clear();
      sends.clear();
   }
   
   template<class C>
   bool Stencil<C>::calcRemoteUpdateSendsAndReceives() {
      if (initialized == false) return false;
      bool success = true;
      clear();
    
      return success;
   }
   
   template<class C>
   bool Stencil<C>::calcLocalUpdateSendsAndReceives() {
      if (initialized == false) return false;
      bool success = true;
      clear();

      std::vector<CellID> localCells;
      parGrid->getLocalCellIDs(localCells);
      for (size_t i=0; i<localCells.size(); ++i) {
	 unsigned int N_remoteNeighbours = 0;
	 const std::vector<CellID> nbrIDs = parGrid->getCellNeighbourIDs(localCells[i]);
	 const std::vector<MPI_processID> nbrHosts = parGrid->getCellNeighbourHosts(localCells[i]);
	 for (size_t nbr=0; nbr<nbrIDs.size(); ++nbr) {
	    const CellID nbrID = nbrIDs[nbr];
	    // Check that neighbour exists and is not local:
	    if (nbrID == parGrid->invalid()) continue;
	    if (nbrHosts[nbr] == parGrid->getRank()) continue;
	    
	    // If neighbour type ID is in sentNbrTypeIDs, add a send.
	    if (std::find(sentNbrTypeIDs.begin(),sentNbrTypeIDs.end(),nbr) != sentNbrTypeIDs.end())
	      sends[nbrHosts[nbr]].insert(localCells[i]);
	    // If neighbour type ID is in receivedNbrTypeIDs, add a receive:
	    if (std::find(receivedNbrTypeIDs.begin(),receivedNbrTypeIDs.end(),nbr) != receivedNbrTypeIDs.end()) {
	       recvs[nbrHosts[nbr]].insert(nbrID);
	       ++N_remoteNeighbours;
	    }
	 }
	    
	 if (N_remoteNeighbours == 0) innerCells.push_back(localCells[i]);
	 else boundaryCells.push_back(localCells[i]);
      }
      return success;
   }

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
      for (typename std::map<MPI_processID,TypeCache>::iterator jt=it->second.begin(); jt!=it->second.end(); ++jt) {
	 MPI_Datatype type;
	 const size_t N_recvs = recvs.find(jt->first)->second.size();
	 const size_t N_sends = sends.find(jt->first)->second.size();
	 blockLengths = new int[std::max(N_recvs,N_sends)];
	 displacements = new MPI_Aint[std::max(N_recvs,N_sends)];
	    
	 // Get displacements from cells receiving data:
	 int counter = 0;
	 for (std::set<CellID>::const_iterator i=recvs[jt->first].begin(); i!=recvs[jt->first].end(); ++i) {
	    const CellID cellID = *i;
	    (*parGrid)[cellID]->getData(false,it->first,blockLengths+counter,displacements+counter,&type);
	    ++counter;
	 }
	    
	 // Create MPI datatype for receiving all data at once from process jt->first.
	 // sum_receives is the total number of datatypes committed (summed over all processes):
	 jt->second.recvs.push_back(TypeWrapper());
	 MPI_Type_create_hindexed(N_recvs,blockLengths,displacements,type,&(jt->second.recvs.back().type));
	 MPI_Type_commit(&(jt->second.recvs.back().type));
	 ++info->second.N_receives;
	    
	 // Get displacements from cells sending data:
	 counter = 0;
	 for (std::set<CellID>::const_iterator i=sends[jt->first].begin(); i!=sends[jt->first].end(); ++i) {
	    const CellID cellID = *i;
	    (*parGrid)[cellID]->getData(false,it->first,blockLengths+counter,displacements+counter,&type);
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
      bool success = false;
      switch (stencilType) {
       case localToRemoteUpdates:
	 success = calcLocalUpdateSendsAndReceives();
	 break;
       case remoteToLocalUpdates:
	 success = calcRemoteUpdateSendsAndReceives();
	 break;
      }

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
      currentCellID = invalid();
      currentCell = localCells.end();
      recalculateInteriorCells = true;
      recalculateExteriorCells = true;
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
      
      // Insert a new cell to localCells. The insertion will fail 
      // if the cell already exists in localCells. Readers-writers 
      // lock will serialize access to localCells:
      std::pair<typename std::map<CellID,ParCell<C> >::iterator,bool> result;
      result = localCells.insert(std::make_pair(cellID,ParCell<C>()));
      
      // Check that the cell was inserted successfully:
      if (result.second == false) return false;
      
      // Copy cell's neighbours and increase reference count to cell's neighbours, in
      // multithreaded mode this contains readers-writers lock
      // (inside addAndIncreaseNeighbourReferenceEntry):
      (result.first)->second.host = getRank();
      (result.first)->second.info[0] = 27;
      (result.first)->second.neighbours.resize(27);
      (result.first)->second.neighbourTypes.resize(27);
      for (size_t n=0; n<27; ++n) {
	 (result.first)->second.neighbours[n] = invalid();
	 (result.first)->second.neighbourTypes[n] = n;
      }
      (result.first)->second.neighbourFlags = (1 << calcNeighbourTypeID(0,0,0));
      for (size_t n=0; n<nbrIDs.size(); ++n) {
	 (result.first)->second.neighbourFlags = ((result.first)->second.neighbourFlags | (1 << nbrTypes[n]));
	 (result.first)->second.neighbours[nbrTypes[n]] = nbrIDs[n];
	 addAndIncreaseNeighbourReferenceEntry(nbrIDs[n]);
      }
      (result.first)->second.neighbours[13] = cellID;
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
      
      // Update remote neighbour hosts:
      if (syncCellHosts() == false) {
	 std::cerr << "PARGRID FATAL ERROR: sync cell hosts failed!" << std::endl;
	 return false;
      }
      
      // Calculate neighbour processes:
      nbrProcesses.clear();
      for (typename std::map<CellID,ParCell<C> >::const_iterator it=remoteCells.begin(); it!=remoteCells.end(); ++it) {
	 nbrProcesses.insert(it->second.host);
      }

      // Check that data from user is ok:
      int successSum = 0;
      int mySuccess = 0;
      if (checkInternalStructures() == false) ++mySuccess;      
      MPI_Allreduce(&mySuccess,&successSum,1,MPI_Type<int>(),MPI_SUM,comm);
      if (successSum > 0) return false;
      
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

   template<class C>
   std::map<CellID,int>::iterator ParGrid<C>::addAndIncreaseNeighbourReferenceEntry(CellID nbrID) {
      std::map<CellID,int>::iterator it;
      
      // Search if an entry already exists in nbrReferences. 
      // If one does, increase its reference count.
      it = nbrReferences.find(nbrID);
      if (it == nbrReferences.end()) {
	 std::pair<std::map<CellID,int>::iterator,bool> result;
	 result = nbrReferences.insert(std::make_pair(nbrID,0));
	 it = result.first;
      }
      ++(it->second);
      return it;
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
      bool success = true;
      
      // Request load balance from Zoltan, and get cells which should be imported and exported. 
      // NOTE that import/export lists may contain cells that already are on this process, at
      // least with RANDOM load balancing method!
      int changes,N_globalIDs,N_localIDs,N_import,N_export;
      int* importProcesses;
      int* importParts;
      int* exportProcesses;
      int* exportParts;
      ZOLTAN_ID_PTR importGlobalIDs;
      ZOLTAN_ID_PTR importLocalIDs;
      ZOLTAN_ID_PTR exportGlobalIDs;
      ZOLTAN_ID_PTR exportLocalIDs;
      if (zoltan->LB_Partition(changes,N_globalIDs,N_localIDs,N_import,importGlobalIDs,importLocalIDs,
			       importProcesses,importParts,N_export,exportGlobalIDs,exportLocalIDs,
			       exportProcesses,exportParts) != ZOLTAN_OK) {
	 std::cerr << "ParGrid FATAL ERROR: Zoltan failed on load balancing!" << std::endl << std::flush;
	 zoltan->LB_Free_Part(&importGlobalIDs,&importLocalIDs,&importProcesses,&importParts);
	 zoltan->LB_Free_Part(&exportGlobalIDs,&exportLocalIDs,&exportProcesses,&exportParts);
	 success = false;
	 return success;
      }

      // We need to exchange lists of migrating cells and their new hosts with neighbouring process. 
      // A process is a neighbouring process if it has at least one of my local cells' (remote) neighbours.
      /*
      nbrProcesses.clear();
      for (typename std::map<CellID,ParCell<C> >::const_iterator it=remoteCells.begin(); it!=remoteCells.end(); ++it) {
	 nbrProcesses.insert(it->second.host);
      }*/

      // Update localCells based on information available to this process.
      // This needs to occur after neighbouringHosts has been updated, otherwise we 
      // may get wrong neighbouring hosts. Note that localCells[importGlobalIDs[i]] 
      // allocates memory for new local cells:
      std::set<MPI_processID> exportProcs; // List of processes this process exports cells to
      std::set<MPI_processID> importProcs; // List of processes this process imports cells from
      for (int i=0; i<N_export; ++i) {
	 localCells[exportGlobalIDs[i]].host = exportProcesses[i]; 
	 if (exportProcesses[i] != myrank) exportProcs.insert(exportProcesses[i]);
      }
      for (int i=0; i<N_import; ++i) {
	 localCells[importGlobalIDs[i]].host = myrank; 
	 if (importProcesses[i] != myrank) importProcs.insert(importProcesses[i]);
      }
      
      // Allocate enough MPI Requests for sends and receives used in cell migration:
      size_t recvSize = std::max(2*nbrProcesses.size(),static_cast<size_t>(N_import));
      size_t sendSize = std::max(2*nbrProcesses.size(),static_cast<size_t>(N_export));
      recvSize = std::max(2*importProcs.size(),recvSize);
      sendSize = std::max(2*exportProcs.size(),sendSize);
      recvRequests.resize(recvSize);
      sendRequests.resize(sendSize);
      
      // Allocate arrays for exchanging cell migrations with neighbouring processes:
      int* neighbourChanges = new int[nbrProcesses.size()];                                   // Number of exports nbrs. process has
      ZOLTAN_ID_TYPE** neighbourMigratingCellIDs = new ZOLTAN_ID_TYPE* [nbrProcesses.size()]; // Global IDs of cells nbr. process is exporting
      MPI_processID** neighbourMigratingHosts    = new MPI_processID* [nbrProcesses.size()];  // New hosts for cells nbr. process exports
      
      // Send the number of cells this process is exporting to all neighbouring process, 
      // and receive the number of exported cells per neighbouring process:
      size_t counter = 0;
      for (std::set<MPI_processID>::const_iterator it=nbrProcesses.begin(); it!=nbrProcesses.end(); ++it) {
	 MPI_Irecv(&(neighbourChanges[counter]),1,MPI_INT,*it,   *it,comm,&(recvRequests[counter]));
	 MPI_Isend(&N_export,                   1,MPI_INT,*it,myrank,comm,&(sendRequests[counter]));
	 ++counter;
      }
      // Wait for information to arrive from neighbours:
      MPI_Waitall(nbrProcesses.size(),&(recvRequests[0]),MPI_STATUSES_IGNORE);
      MPI_Waitall(nbrProcesses.size(),&(sendRequests[0]),MPI_STATUSES_IGNORE);
      
      // Allocate arrays for receiving migrating cell IDs and new hosts
      // from neighbouring processes. Exchange data with neighbours:
      counter = 0;
      for (std::set<MPI_processID>::const_iterator it=nbrProcesses.begin(); it!=nbrProcesses.end(); ++it) {
	 neighbourMigratingCellIDs[counter] = new ZOLTAN_ID_TYPE[neighbourChanges[counter]];
	 neighbourMigratingHosts[counter] = new MPI_processID[neighbourChanges[counter]];
	 
	 MPI_Irecv(neighbourMigratingCellIDs[counter],neighbourChanges[counter],MPI_Type<ZOLTAN_ID_TYPE>(),*it,*it,comm,&(recvRequests[2*counter+0]));
	 MPI_Irecv(neighbourMigratingHosts[counter]  ,neighbourChanges[counter],MPI_Type<MPI_processID>() ,*it,*it,comm,&(recvRequests[2*counter+1]));
	 MPI_Isend(exportGlobalIDs,N_export,MPI_Type<ZOLTAN_ID_TYPE>(),*it,myrank,comm,&(sendRequests[2*counter+0]));
	 MPI_Isend(exportProcesses,N_export,MPI_Type<MPI_processID>() ,*it,myrank,comm,&(sendRequests[2*counter+1]));
	 ++counter;
      }
      MPI_Waitall(2*nbrProcesses.size(),&(recvRequests[0]),MPI_STATUSES_IGNORE);
      MPI_Waitall(2*nbrProcesses.size(),&(sendRequests[0]),MPI_STATUSES_IGNORE);
      
      // Do a second pass of local updates based on information received from neighbours. 
      // We only need to update map remoteCells, because remote processes cannot migrate 
      // local cells of this process:
      counter = 0;
      typename std::map<CellID,ParCell<C> >::iterator remoteCellIt;
      for (std::set<MPI_processID>::const_iterator it=nbrProcesses.begin(); it!=nbrProcesses.end(); ++it) {
	 for (int i=0; i<neighbourChanges[counter]; ++i) {
	    remoteCellIt = remoteCells.find(neighbourMigratingCellIDs[counter][i]);
	    if (remoteCellIt != remoteCells.end()) {
	       remoteCellIt->second.host = neighbourMigratingHosts[counter][i];
	       //std::cerr << "P#" << myrank << " changed rem C#" << neighbourMigratingCellIDs[counter][i] << " host to P#" << remoteCellIt->second.host;
	       //std::cerr << " from P#" << *it << std::endl;
	    }
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
      
      // ********************************************************* //
      // ***** SEND EXPORTED CELLS AND RECEIVE IMPORTED ONES ***** //
      // ********************************************************* //
      
      // Get the number of metadata elements user needs for cell migration.
      // All cells are required to use the same amount of metadata elements, 
      // so we only need to ask this from one of the local cells:
      unsigned int N_metadataElements = localCells.begin()->second.getMetadataElements(0);

      // Allocate arrays for MPI datatype that contains metadata needed in migration:
      MPI_Datatype MPIdataType;
      //MPI_Datatype* datatypes = new MPI_Datatype[N_metadataElements];
      //int* blockLengths       = new int[N_metadataElements];
      //MPI_Aint* displacements = new MPI_Aint[N_metadataElements];
      
      // ********** NEEDS IMPROVEMENT (CRITICAL) --- ONLY ONE SEND/RECV PER NBR PROCESS **********

      // Count the number of cells exported per neighbouring process, and number of cells 
      // imported per neighbouring process. All data needs to be sent (and received) with a 
      // single MPI call per remote neighbour, otherwise unexpected message buffers may run out!
      std::map<MPI_processID,unsigned int> exportsPerProcess;
      std::map<MPI_processID,unsigned int> importsPerProcess;
      for (int i=0; i<N_import; ++i) ++importsPerProcess[importProcesses[i]];
      for (int i=0; i<N_export; ++i) ++exportsPerProcess[exportProcesses[i]];
      /*
      for (std::map<MPI_processID,unsigned int>::iterator it=importsPerProcess.begin(); it!=importsPerProcess.end(); ++it) {
	 std::cerr << "P#" << getRank() << " imports " << it->second << " from P#" << it->first << std::endl;
      }
      for (std::map<MPI_processID,unsigned int>::iterator it=exportsPerProcess.begin(); it!=exportsPerProcess.end(); ++it) {
	 std::cerr << "P#" << getRank() << " exports " << it->second << " to P#" << it->first << std::endl;
      }*/
      
      // Create temporary arrays for metadata incoming from each importing process:
      std::vector<MPI_Datatype*> datatypes(importsPerProcess.size());
      std::vector<int*> blockLengths(importsPerProcess.size());
      std::vector<MPI_Aint*> displacements(importsPerProcess.size());
      std::vector<unsigned int> indices(importsPerProcess.size());
      std::vector<MPI_processID> importNeighbours(importsPerProcess.size());
      counter = 0;
      for (std::map<MPI_processID,unsigned int>::const_iterator it=importsPerProcess.begin(); it!=importsPerProcess.end(); ++it) {
	 datatypes[counter]        = new MPI_Datatype[N_metadataElements*it->second];
	 blockLengths[counter]     = new int[N_metadataElements*it->second];
	 displacements[counter]    = new MPI_Aint[N_metadataElements*it->second];
	 importNeighbours[counter] = it->first;
	 indices[counter] = 0;
	 ++counter;
      }
      std::sort(importNeighbours.begin(),importNeighbours.end());
      /*
      for (size_t i=0; i<importNeighbours.size(); ++i) {
	 std::cerr << "P#" << getRank() << " import nbr " << i << " = " << importNeighbours[i] << std::endl;
      }*/
      
      // Get metadata for imported cells - these were created above. In practice we fetch addresses where 
      // incoming metadata is to be written:
      for (int i=0; i<N_import; ++i) {
	 const int identifier = 0;
	 const MPI_processID procIndex = lower_bound(importNeighbours.begin(),importNeighbours.end(),importProcesses[i]) - importNeighbours.begin();
	 const unsigned int arrayIndex = indices[procIndex];
	 localCells[importGlobalIDs[i]].getMetadata(identifier,blockLengths[procIndex]+arrayIndex,displacements[procIndex]+arrayIndex,datatypes[procIndex]+arrayIndex);
	 indices[procIndex] += N_metadataElements;
      }
      
      // Create an MPI struct containing all data received from importing process it->first and post receive:
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
      
      /*
      // Post receive for each imported cell. Identifier value zero is reserved 
      // for cell migration, i.e. all cell data is to be sent/received:
      for (int i=0; i<N_import; ++i) {
	 const int identifier = 0;
	 const MPI_processID source = importProcesses[i];
	 const int tag              = importProcesses[i];
	 localCells[importGlobalIDs[i]].getMetadata(identifier,blockLengths,displacements,datatypes);
	 MPI_Type_create_struct(N_metadataElements,blockLengths,displacements,datatypes,&MPIdataType);
	 MPI_Type_commit(&MPIdataType);
	 MPI_Irecv(MPI_BOTTOM,1,MPIdataType,source,tag,comm,&(recvRequests[i]));
	 MPI_Type_free(&MPIdataType);
      }*/

      // Allocate arrays for sending metadata to export processes:
      datatypes.resize(exportsPerProcess.size());
      blockLengths.resize(exportsPerProcess.size());
      displacements.resize(exportsPerProcess.size());
      indices.resize(exportsPerProcess.size());
      std::vector<MPI_processID> exportNeighbours(exportsPerProcess.size());
      counter = 0;
      for (std::map<MPI_processID,unsigned int>::const_iterator it=exportsPerProcess.begin(); it!=exportsPerProcess.end(); ++it) {
	 datatypes[counter]        = new MPI_Datatype[N_metadataElements*it->second];
	 blockLengths[counter]     = new int[N_metadataElements*it->second];
	 displacements[counter]    = new MPI_Aint[N_metadataElements*it->second];
	 exportNeighbours[counter] = it->first;
	 indices[counter] = 0;
	 ++counter;
      }
      std::sort(exportNeighbours.begin(),exportNeighbours.end());
      
      // Fetch metadata to send to export processes:
      for (int i=0; i<N_export; ++i) {
	 const int identifier = 0;
	 const MPI_processID procIndex = lower_bound(exportNeighbours.begin(),exportNeighbours.end(),exportProcesses[i]) - exportNeighbours.begin();
	 const unsigned int arrayIndex = indices[procIndex];
	 localCells[exportGlobalIDs[i]].getMetadata(identifier,blockLengths[procIndex]+arrayIndex,displacements[procIndex]+arrayIndex,datatypes[procIndex]+arrayIndex);
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
      
      /*
      // Post send for each exported cell:
      for (int i=0; i<N_export; ++i) {
	 const int identifier = 0;
	 const MPI_processID dest = exportProcesses[i];
	 const int tag            = myrank;
	 localCells[exportGlobalIDs[i]].getMetadata(identifier,blockLengths,displacements,datatypes);
	 MPI_Type_create_struct(N_metadataElements,blockLengths,displacements,datatypes,&MPIdataType);
	 MPI_Type_commit(&MPIdataType);
	 MPI_Isend(MPI_BOTTOM,1,MPIdataType,dest,tag,comm,&(sendRequests[i]));
	 MPI_Type_free(&MPIdataType);
      }*/
      
      // Deallocate arrays used for MPI datatypes in cell migration:
      //delete [] datatypes; datatypes = NULL;
      //delete [] blockLengths; blockLengths = NULL;
      //delete [] displacements; displacements = NULL;
      for (size_t i=0; i<exportsPerProcess.size(); ++i) {
	 delete [] datatypes[i]; datatypes[i] = NULL;
	 delete [] blockLengths[i]; blockLengths[i] = NULL;
	 delete [] displacements[i]; displacements[i] = NULL;
      }
      
      //MPI_Waitall(importsPerProcess.size(),&(recvRequests[0]),MPI_STATUSES_IGNORE);
      //MPI_Waitall(exportsPerProcess.size(),&(sendRequests[0]),MPI_STATUSES_IGNORE);
      
      // ********** END NEEDS IMPROVEMENT **********

      // Allocate arrays for MPI datatypes describing imported cell data:
      unsigned int N_dataElements = localCells.begin()->second.getDataElements(0);
      datatypes.resize(importsPerProcess.size());
      blockLengths.resize(importsPerProcess.size());
      displacements.resize(importsPerProcess.size());
      indices.resize(importsPerProcess.size());
      counter = 0;
      for (std::map<MPI_processID,unsigned int>::const_iterator it=importsPerProcess.begin(); it!=importsPerProcess.end(); ++it) {
	 datatypes[counter]        = new MPI_Datatype[N_dataElements*it->second];
	 blockLengths[counter]     = new int[N_dataElements*it->second];
	 displacements[counter]    = new MPI_Aint[N_dataElements*it->second];
	 indices[counter] = 0;
	 ++counter;
      }
      
      /*
      // Allocate arrays for MPI datatype that contains migrated cell data:
      unsigned int N_dataElements = localCells.begin()->second.getDataElements(0);
      datatypes = new MPI_Datatype[N_dataElements];
      blockLengths = new int[N_dataElements];
      displacements = new MPI_Aint[N_dataElements];
      */
      
      // Wait for metadata sends & receives to complete:
      MPI_Waitall(importsPerProcess.size(),&(recvRequests[0]),MPI_STATUSES_IGNORE);
      MPI_Waitall(exportsPerProcess.size(),&(sendRequests[0]),MPI_STATUSES_IGNORE);

      // Fetch data to arrays describing imported data:
      for (int i=0; i<N_import; ++i) {
	 const bool sending = false;
	 const int identifier = 0;
	 const MPI_processID procIndex = lower_bound(importNeighbours.begin(),importNeighbours.end(),importProcesses[i]) - importNeighbours.begin();
	 const unsigned int arrayIndex = indices[procIndex];
	 localCells[importGlobalIDs[i]].getData(sending,identifier,blockLengths[procIndex]+arrayIndex,displacements[procIndex]+arrayIndex,datatypes[procIndex]+arrayIndex);
	 indices[procIndex] += N_dataElements;
      }
      
      // Post receives for imported data:
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
      
      datatypes.resize(exportsPerProcess.size());
      blockLengths.resize(exportsPerProcess.size());
      displacements.resize(exportsPerProcess.size());
      indices.resize(exportsPerProcess.size());
      counter = 0;
      for (std::map<MPI_processID,unsigned int>::const_iterator it=exportsPerProcess.begin(); it!=exportsPerProcess.end(); ++it) {
	 datatypes[counter]        = new MPI_Datatype[N_dataElements*it->second];
	 blockLengths[counter]     = new int[N_dataElements*it->second];
	 displacements[counter]    = new MPI_Aint[N_dataElements*it->second];
	 indices[counter] = 0;
	 ++counter;
      }
      
      for (int i=0; i<N_export; ++i) {
	 const bool sending = true;
	 const int identifier = 0;
	 const MPI_processID procIndex = lower_bound(exportNeighbours.begin(),exportNeighbours.end(),exportProcesses[i]) - exportNeighbours.begin();
	 const unsigned int arrayIndex = indices[procIndex];
	 localCells[exportGlobalIDs[i]].getData(sending,identifier,blockLengths[procIndex]+arrayIndex,displacements[procIndex]+arrayIndex,datatypes[procIndex]+arrayIndex);
	 indices[procIndex] += N_dataElements;
      }
      
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
      
      for (size_t i=0; i<exportsPerProcess.size(); ++i) {
	 delete [] datatypes[i]; datatypes[i] = NULL;
	 delete [] blockLengths[i]; blockLengths[i] = NULL;
	 delete [] displacements[i]; displacements[i] = NULL;
      }
            
      /*
      // Post receives for imported cell data:
      for (int i=0; i<N_import; ++i) {
	 const bool sending = false;
	 const int identifier = 0;
	 const MPI_processID source = importProcesses[i];
	 const int tag              = importProcesses[i];
	 localCells[importGlobalIDs[i]].getData(sending,identifier,blockLengths,displacements,datatypes);
	 MPI_Type_create_struct(N_dataElements,blockLengths,displacements,datatypes,&MPIdataType);
	 MPI_Type_commit(&MPIdataType);
	 MPI_Irecv(MPI_BOTTOM,1,MPIdataType,source,tag,comm,&(recvRequests[i]));
	 MPI_Type_free(&MPIdataType);
      }
      
      // Post sends for exported cell data:
      for (int i=0; i<N_export; ++i) {
	 const bool sending = true;
	 const int identifier = 0;
	 const MPI_processID dest = exportProcesses[i];
	 const int tag            = myrank;
	 localCells[exportGlobalIDs[i]].getData(sending,identifier,blockLengths,displacements,datatypes);
	 MPI_Type_create_struct(N_dataElements,blockLengths,displacements,datatypes,&MPIdataType);
	 MPI_Type_commit(&MPIdataType);
	 MPI_Isend(MPI_BOTTOM,1,MPIdataType,dest,tag,comm,&(sendRequests[i]));
	 MPI_Type_free(&MPIdataType);
      }

      // Deallocate arrays for MPI datatype in cell migration:
      delete [] datatypes; datatypes = NULL;
      delete [] blockLengths; blockLengths = NULL;
      delete [] displacements; displacements = NULL;
      */
      
      std::map<MPI_processID,std::set<std::pair<CellID,MPI_processID> > > hostUpdates;
      
      // Remove exported cell's reference counts. If a remote cell ends 
      // up having zero references, remove it from remoteCells:
      for (int i=0; i<N_export; ++i) {
	 const MPI_processID newHost = localCells[exportGlobalIDs[i]].host;
	 std::map<CellID,int>::iterator it;
	 for (size_t j=0; j<localCells[exportGlobalIDs[i]].neighbours.size(); ++j) {
	    const CellID nbrID = localCells[exportGlobalIDs[i]].neighbours[j];
	    if (j == 13 || nbrID == invalid()) continue;
	    
	    // Find the neighbour's host and insert into update list, if necessary:
	    typename std::map<CellID,ParCell<C> >::const_iterator nbrIt = localCells.find(nbrID);
	    if (nbrIt == localCells.end()) nbrIt = remoteCells.find(nbrID);
	    const MPI_processID nbrHost = nbrIt->second.host;	    
	    if (nbrHost != newHost) hostUpdates[newHost].insert(std::make_pair(nbrID,nbrHost));
	    
	    // Reduce neighbours reference count and remove cell if necessary:
	    it = nbrReferences.find(localCells[exportGlobalIDs[i]].neighbours[j]);
	    --it->second;
	    if (it->second == 0) {
	       remoteCells.erase(it->first);
	       nbrReferences.erase(it);
	    }
	 }
      }
      
      // Erase imported cells from remoteCells, this is required in cases 
      // where a remote cell became a local cell. These cells were inserted 
      // to localCells above:
      for (int i=0; i<N_import; ++i) {
	 if (remoteCells.find(importGlobalIDs[i]) != remoteCells.end()) {
	    remoteCells.erase(importGlobalIDs[i]);
	 }
      }
      
      /* TEST
      // Wait for imported cell data to arrive:
      MPI_Waitall(N_import,&(recvRequests[0]),MPI_STATUSES_IGNORE);
      END TEST (REMOVE COMMENTS) */
      MPI_Waitall(importsPerProcess.size(),&(recvRequests[0]),MPI_STATUSES_IGNORE);
      
      // Insert imported cell's remote neighbours and increase their reference counts. 
      // Also calculate neighbour flags for imported cells:
      for (int i=0; i<N_import; ++i) {
	 typename std::map<CellID,ParCell<C> >::iterator it = localCells.find(importGlobalIDs[i]);
	 it->second.neighbourFlags = 0;
	 for (size_t j=0; j<it->second.neighbours.size(); ++j) {
	    if (it->second.neighbours[j] == invalid()) continue;
	    it->second.neighbourFlags = (it->second.neighbourFlags | (1 << j));
	    if (j == 13) continue;

	    addAndIncreaseNeighbourReferenceEntry(it->second.neighbours[j]);
	    if (localCells.find(it->second.neighbours[j]) == localCells.end()) {
	       remoteCells[it->second.neighbours[j]];
	    }
	 }
      }
      
      /* TEST
      // Wait for exported cell data sends to complete:
      MPI_Waitall(N_export,&(sendRequests[0]),MPI_STATUSES_IGNORE);
      END TEST (REMOVE COMMENTS) */
      MPI_Waitall(exportsPerProcess.size(),&(sendRequests[0]),MPI_STATUSES_IGNORE);
      
      // Erase exported cell from localCells if the cell was sent to another process. 
      // If the localCell has neighbour references remaining, insert it to remoteCells
      // (a local cell became a remote cell). This can be done only after exported 
      // cell data has been sent:
      for (int i=0; i<N_export; ++i) {
	 if (exportProcesses[i] == getRank()) continue;
	 localCells.erase(exportGlobalIDs[i]);
	 std::map<CellID,int>::const_iterator it = nbrReferences.find(exportGlobalIDs[i]);
	 if (it == nbrReferences.end()) continue;
	 remoteCells[exportGlobalIDs[i]].host = exportProcesses[i];
      }

      // Send cell host update list to every unique process in exportProcesses, 
      // and receive a list from processes in importProcesses. First we need to 
      // receive the number of incoming updates from each importint process:
      counter = 0;
      size_t* incomingUpdates = new size_t[importProcs.size()];
      size_t* outgoingUpdates = new size_t[exportProcs.size()];
      for (std::set<MPI_processID>::const_iterator it=importProcs.begin(); it!=importProcs.end(); ++it) {
	 MPI_Irecv(incomingUpdates+counter,1,MPI_Type<size_t>(),*it,*it,comm,&(recvRequests[counter]));
	 ++counter;
      }
      counter = 0;
      for (std::set<MPI_processID>::const_iterator it=exportProcs.begin(); it!=exportProcs.end(); ++it) {
	 outgoingUpdates[counter] = hostUpdates[*it].size();
	 MPI_Isend(outgoingUpdates+counter,1,MPI_Type<size_t>(),*it,myrank,comm,&(sendRequests[counter]));
	 ++counter;
      }

      // Allocate buffers for sending host updates while we are waiting 
      // for MPI transfers to complete:
      CellID** incomingCellIDs      = new CellID* [importProcs.size()];
      MPI_processID** incomingHosts = new MPI_processID* [importProcs.size()];
      CellID** outgoingCellIDs      = new CellID* [exportProcs.size()];
      MPI_processID** outgoingHosts = new MPI_processID* [exportProcs.size()];
      
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

      // Wait for numbers of updates per importing host to arrive:
      MPI_Waitall(importProcs.size(),&(recvRequests[0]),MPI_STATUSES_IGNORE);

      // Allocate buffers for incoming cell updates:
      for (size_t i=0; i<importProcs.size(); ++i) {
	 incomingCellIDs[i] = new CellID[incomingUpdates[i]];
	 incomingHosts[i]   = new MPI_processID[incomingUpdates[i]];
      }

      // Wait for sends to complete:
      MPI_Waitall(exportProcs.size(),&(sendRequests[0]),MPI_STATUSES_IGNORE);

      // Receive host updates from neighbours:
      counter = 0;
      for (std::set<MPI_processID>::const_iterator it=importProcs.begin(); it!=importProcs.end(); ++it) {
	 MPI_Irecv(incomingCellIDs[counter],incomingUpdates[counter],MPI_Type<CellID>()       ,*it,*it,comm,&(recvRequests[2*counter+0]));
	 MPI_Irecv(incomingHosts[counter]  ,incomingUpdates[counter],MPI_Type<MPI_processID>(),*it,*it,comm,&(recvRequests[2*counter+1]));
	 ++counter;
      }
      // Send my updates:
      counter = 0;
      for (std::set<MPI_processID>::const_iterator it=exportProcs.begin(); it!=exportProcs.end(); ++it) {
	 MPI_Isend(outgoingCellIDs[counter],outgoingUpdates[counter],MPI_Type<CellID>()       ,*it,myrank,comm,&(sendRequests[2*counter+0]));
	 MPI_Isend(outgoingHosts[counter]  ,outgoingUpdates[counter],MPI_Type<MPI_processID>(),*it,myrank,comm,&(sendRequests[2*counter+1]));
	 ++counter;
      }

      // Wait for incoming host updates:
      MPI_Waitall(2*importProcs.size(),&(recvRequests[0]),MPI_STATUSES_IGNORE);
      
      // Update remote cell hosts based on received information:
      counter = 0;
      for (std::set<MPI_processID>::const_iterator it=importProcs.begin(); it!=importProcs.end(); ++it) {
	 typename std::map<CellID,ParCell<C> >::iterator remIt;
	 for (size_t i=0; i<incomingUpdates[counter]; ++i) {
	    remIt = remoteCells.find(incomingCellIDs[counter][i]);
	    if (remIt != remoteCells.end()) remIt->second.host = incomingHosts[counter][i];
	 }
	 ++counter;
      }
      
      // Wait for outgoing updates:
      MPI_Waitall(2*exportProcs.size(),&(sendRequests[0]),MPI_STATUSES_IGNORE);

      // Deallocate arrays used in host updates:
      delete [] incomingUpdates; incomingUpdates = NULL;
      delete [] outgoingUpdates; outgoingUpdates = NULL;
      
      // Deallocate arrays used in host updates:
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

      // Calculate neighbour processes:
      nbrProcesses.clear();
      for (typename std::map<CellID,ParCell<C> >::const_iterator it=remoteCells.begin(); it!=remoteCells.end(); ++it) {
	 nbrProcesses.insert(it->second.host);
      }
      
      // FIXME
      for (typename std::map<CellID,ParCell<C> >::iterator it=localCells.begin(); it!=localCells.end(); ++it) {
	 for (size_t n=0; n<27; ++n) {
	    if (it->second.neighbours[n] == invalid()) it->second.neighbourHosts[n] = MPI_PROC_NULL;
	    const CellID nbrID = it->second.neighbours[n];
	    typename std::map<CellID,ParCell<C> >::const_iterator jt = localCells.find(nbrID);
	    if (jt == localCells.end()) jt = remoteCells.find(nbrID);
	    it->second.neighbourHosts[n] = jt->second.host;
	 }
      }
      // END FIXME

      // Invalidate current cell iterator:
      invalidate();
      checkInternalStructures(); // DEBUG
      return success;
   }
   
   /** Synchronize MPI processes in the communicator ParGrid is using.
    * Only the master thread blocks, all other threads will exit this 
    * function immediately.
    */
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
      
      // Count neighbour references, and collect global IDs of remote neighbours. 
      // Also check neighbourFlags for correctness:
      for (typename std::map<CellID,ParCell<C> >::const_iterator it=localCells.begin(); it!=localCells.end(); ++it) {
	 for (size_t i=0; i<it->second.neighbours.size(); ++i) {
	    if (it->second.neighbours[i] == invalid()) {
	       if (((it->second.neighbourFlags >> i) & 1) != 0) {
		  std::cerr << "P#" << myrank << " C#" << it->first << " nbrFlag is one for non-existing nbr type " << i << std::endl;
	       }
	       continue;
	    }
	    if (((it->second.neighbourFlags >> i) & 1) != 1) {
	       std::cerr << "P#" << myrank << " C#" << it->first << " nbrFlag is zero for existing nbr type " << i << std::endl;
	    }
	    if (i == 13) continue;

	    ++tmpReferences[it->second.neighbours[i]];
	    if (localCells.find(it->second.neighbours[i]) == localCells.end()) tmpRemoteCells.insert(it->second.neighbours[i]);
	 }
      }
      // Check that nbrReferences has all neighbours with correct reference count:
      for (std::map<CellID,int>::const_iterator it=tmpReferences.begin(); it!=tmpReferences.end(); ++it) {
	 std::map<CellID,int>::const_iterator jt=nbrReferences.find(it->first);
	 if (jt == nbrReferences.end()) {
	    std::cerr << "P#" << myrank << " nbrRefs does not contain C#" << it->first << std::endl; 
	    success = false;
	 } else if (it->second != jt->second) {
	    std::cerr << "P#" << myrank << " nbrRefs C#" << it->first << " should be " << it->second << " is " << jt->second << std::endl;
	    success = false;
	 }
      }
      // Check that nbrReferences does not have entries for non-existing cells:
      for (std::map<CellID,int>::const_iterator it=nbrReferences.begin(); it!=nbrReferences.end(); ++it) {
	 std::map<CellID,int>::const_iterator jt=tmpReferences.find(it->first);
	 if (jt == tmpReferences.end()) {
	    std::cerr << "P#" << myrank << " nbrRefs entry for C#" << it->first << " value " << it->second << " should not exist" << std::endl;
	    success = false;
	 }
      }
      
      // Assume that localCells is correct. Check that remoteCells is correct.
      for (std::set<CellID>::const_iterator it=tmpRemoteCells.begin(); it!=tmpRemoteCells.end(); ++it) {
	 typename std::map<CellID,ParCell<C> >::const_iterator jt=remoteCells.find(*it);
	 if (jt == remoteCells.end()) {
	    std::cerr << "P#" << myrank << " C#" << *it << " not found from remoteCells!" << std::endl;
	    success = false;
	 }
      }
      // Check that remoteCells does not contain unnecessary entries:
      for (typename std::map<CellID,ParCell<C> >::const_iterator it=remoteCells.begin(); it!=remoteCells.end(); ++it) {
	 std::set<CellID>::const_iterator jt=tmpRemoteCells.find(it->first);
	 if (jt == tmpRemoteCells.end()) {
	    std::cerr << "P#" << myrank << " remoteCells has unnecessary entry for C#" << it->first << std::endl;
	    success = false;
	 }
      }
      // Check that localCells and remoteCells do not contain duplicate cells:
      for (typename std::map<CellID,ParCell<C> >::const_iterator it=localCells.begin(); it!=localCells.end(); ++it) {
	 if (remoteCells.find(it->first) != remoteCells.end()) {
	    std::cerr << "P#" << myrank << " localCell C#" << it->first << " is in remoteCells" << std::endl;
	    success = false;
	 }
      }
      for (typename std::map<CellID,ParCell<C> >::const_iterator it=remoteCells.begin(); it!=remoteCells.end(); ++it) {
	 if (localCells.find(it->first) != localCells.end()) {
	    std::cerr << "P#" << myrank << " remoteCell C#" << it->first << " is in localCells" << std::endl;
	    success = false;
	 }
      }

      // Check that all hosts are valid:
      for (typename std::map<CellID,ParCell<C> >::const_iterator it=localCells.begin(); it!=localCells.end(); ++it) {
	 // Check that all local cells have this process as their host:
	 if (it->second.host != myrank) {
	    std::cerr << "P#" << myrank << " local C#" << it->first << " host " << it->second.host << std::endl;
	    success = false;
	 }
	 // Check that all neighbour hosts have reasonable values:
	 for (size_t i=0; i<it->second.neighbours.size(); ++i) {
	    if (it->second.neighbours[i] == invalid()) continue; // Skip non-existing nbrs
	    if (it->second.neighbourHosts[i] >= getProcesses()) {
	       std::cerr << "P#" << myrank << " local C#" << it->first << " nbr #" << it->second.neighbours[i] << " host: ";
	       std::cerr << it->second.neighbourHosts[i] << " is invalid" << std::endl;
	       success = false;
	    }
	 }
      }
      
      // Check that all remote cells have a correct host. Each process sends everyone else
      // a list of cells it owns. Hosts can be checked based on this information:
      for (MPI_processID p = 0; p < getProcesses(); ++p) {
	 CellID N_cells;
	 if (p == getRank()) {
	    // Tell everyone how many cells this process has:
	    N_cells = localCells.size();
	    MPI_Bcast(&N_cells,1,MPI_Type<CellID>(),p,comm);
	    // Create an array containing cell IDs and broadcast:
	    CellID counter = 0;
	    CellID* cells = new CellID[N_cells];
	    for (typename std::map<CellID,ParCell<C> >::const_iterator c=localCells.begin(); c!=localCells.end(); ++c) {
	       cells[counter] = c->first;
	       ++counter;
	    }
	    MPI_Bcast(cells,N_cells,MPI_Type<CellID>(),p,comm);
	    delete [] cells; cells = NULL;
	 } else {
	    // Receive number of cells process p is sending:
	    MPI_Bcast(&N_cells,1,MPI_Type<CellID>(),p,comm);
	    // Allocate array for receiving cell IDs and receive:
	    CellID* cells = new CellID[N_cells];
	    MPI_Bcast(cells,N_cells,MPI_Type<CellID>(),p,comm);
	    
	    // Check that received cells are not in localCells:
	    for (CellID c=0; c<N_cells; ++c) {
	       if (localCells.find(cells[c]) == localCells.end()) continue;
	       std::cerr << "P#" << myrank << " C#" << cells[c] << " from P#" << p << " is in my localCells!" << std::endl;
	       success = false;
	    }
	    
	    // Check that remoteCells has the correct host:
	    typename std::map<CellID,ParCell<C> >::const_iterator it;
	    for (CellID c=0; c<N_cells; ++c) {
	       it = remoteCells.find(cells[c]);
	       if (it == remoteCells.end()) continue;
	       if (it->second.host == p) continue;
	       std::cerr << "P#" << myrank << " rem C#" << cells[c] << " has host " << it->second.host << " should be " << p << std::endl;
	       success = false;
	    }
	    delete [] cells; cells = NULL;
	 }
      }	    
      return success;
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
      if (it == stencils.end()) {
	 std::cerr << "(PARGRID) ERROR: Non-existing stencil " << stencil << " requested in getBoundaryCells!" << std::endl;
	 exit(1);
      }
      return it->second.getBoundaryCells();
   }
   
   /** Get cell's neighbours host processes. Non-existing neighbours have their
    * host ranks set to value MPI_PROC_NULL.
    * @param cellID Global ID of cell whose neighbours are requested.
    * @return Reference to vector containing neighbours' hosts' MPI ranks.
    * Size of vector is always 27, and it can be indexed with ParGrid::calcNeighbourTypeID.
    */
   template<class C>
   std::vector<MPI_processID>& ParGrid<C>::getCellNeighbourHosts(CellID cellID) {
      if (cellID != currentCellID) {
	 currentCell = localCells.find(cellID);
	 currentCellID = cellID;
      }
      return currentCell->second.neighbourHosts;
   }
   
   /** Get cell's neighbours. Non-existing neighbours have their global IDs 
    * set to value ParGrid::invalid().
    * @param cellID Global ID of cell whose neighbours are requested.
    * @param Reference to vector containing neihbours' global IDs. Size 
    * of vector is always 27. Vector can be indexed with ParGrid::calcNeighbourTypeID.
    */
   template<class C>
   std::vector<CellID>& ParGrid<C>::getCellNeighbourIDs(CellID cellID) {
      if (cellID != currentCellID) {
	 currentCell = localCells.find(cellID);
	 currentCellID = cellID;
      }
      return currentCell->second.neighbours;
   }
   
   template<class C>
   MPI_Comm ParGrid<C>::getComm() const {return comm;}
   
   template<class C>
   std::vector<CellID>& ParGrid<C>::getExteriorCells() {
      if (recalculateExteriorCells == true) {
	 const unsigned int ALL_EXIST = 134217728 - 1; // This value is 2^27 - 1, i.e. integer with first 27 bits flipped
	 exteriorCells.clear();
	 for (typename std::map<CellID,ParCell<C> >::const_iterator it=localCells.begin(); it!=localCells.end(); ++it) {
	    if (it->second.neighbourFlags != ALL_EXIST) {
	       exteriorCells.push_back(it->first);
	    }
	 }
	 recalculateExteriorCells = false;
      }
      return exteriorCells;
   }
   
   /** Query if ParGrid has initialized correctly.
    * The value returned by this function is set in ParGrid::initialize.
    * This function does not contain thread synchronization.
    * @return If true, ParGrid is ready for use.
    */
   template<class C>
   bool ParGrid<C>::getInitialized() const {return initialized;}

   template<class C>
   std::vector<CellID>& ParGrid<C>::getInteriorCells() {
      if (recalculateInteriorCells == true) {
	 const unsigned int ALL_EXIST = 134217728 - 1; // This value is 2^27 - 1, i.e. integer with first 27 bits flipped
	 interiorCells.clear();
	 for (typename std::map<CellID,ParCell<C> >::const_iterator it=localCells.begin(); it!=localCells.end(); ++it) {
	    if (it->second.neighbourFlags == ALL_EXIST) interiorCells.push_back(it->first);
	 }
	 recalculateInteriorCells = false;
      }
      return interiorCells;
   }
   
   /** Get a list of cells stored on this process.
    * @param cells A vector in which global IDs of local cells are copied.
    */
   template<class C>
   void ParGrid<C>::getLocalCellIDs(std::vector<CellID>& cells) const {
      // Allocate enough storage for all existing local cells:
      cells.clear();
      cells.reserve(localCells.size());
      
      // Copy local cell global IDs to output vector:
      for (typename std::map<CellID,ParCell<C> >::const_iterator it=localCells.begin(); it!=localCells.end(); ++it) {
	 cells.push_back(it->first);
      }
   }
   
   /** Get a list of neighbour processes. A process is considered to be a neighbour 
    * if it has one or more this process' local cells' neighbours.
    * @return List of neighbour process IDs.
    */
   template<class C>
   const std::set<MPI_processID>& ParGrid<C>::getNeighbourProcesses() const {return nbrProcesses;}
   
   template<class C>
   int ParGrid<C>::getNeighbourReferenceCount(CellID cellID) {
      std::map<CellID,int>::const_iterator it;
      it = nbrReferences.find(cellID);
      
      if (it == nbrReferences.end()) return -1;
      return it->second;
   }
   
   /** Get the number of cells on this process.
    * @return Number of local cells.
    */
   template<class C>
   CellID ParGrid<C>::getNumberOfLocalCells() const {return localCells.size();}
   
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

   template<class C>
   void ParGrid<C>::getRemoteCellIDs(std::vector<CellID>& cells) const {
      cells.clear();
      cells.reserve(remoteCells.size());
      for (typename std::map<CellID,ParCell<C> >::const_iterator it=remoteCells.begin(); it!=remoteCells.end(); ++it) {
	 cells.push_back(it->first);
      }
   }
   
   /** Get the remote neighbours of a given local cell.
    * @param cellID Global ID of the cell.
    * @param nbrTypeIDs Searched neighours.
    * @param nbrIDs Global IDs of searched remote neighbours are written here.
    * @param hosts MPI host processes of searched remote neighbours are written here.
    */
   template<class C>
   bool ParGrid<C>::getRemoteNeighbours(CellID cellID,const std::vector<unsigned char>& nbrTypeIDs,std::vector<CellID>& nbrIDs,std::vector<MPI_processID>& hosts) {
      // Attempt to find the queried cell:
      nbrIDs.clear();
      hosts.clear();
      typename std::map<CellID,ParCell<C> >::const_iterator it = localCells.find(cellID);
      if (it == localCells.end()) return false;

      // Iterate over given neighbour type IDs and check if this cell has 
      // those neighbours, and if those neighbours are remote:
      MPI_processID host;
      std::map<unsigned char,CellID>::const_iterator nbr;
      typename std::map<CellID,ParCell<C> >::const_iterator jt;
      for (std::vector<unsigned char>::const_iterator nbrType=nbrTypeIDs.begin(); nbrType!=nbrTypeIDs.end(); ++nbrType) {
	 // Attempt to find the neighbour:
	 nbr = it->second.neighbours.find(*nbrType);
	 if (nbr == it->second.neighbours.end()) continue;
	 
	 // Check if the neighbour is local:
	 jt = localCells.find(nbr->second);
	 if (jt == localCells.end()) {
	    jt = remoteCells.find(nbr->second);
	 }
	 if (jt->second.host == getRank()) continue;
	 
	 // Add remote neighbour ID and host to output vectors:
	 nbrIDs.push_back(nbr->second);
	 hosts.push_back(jt->second.host);
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
	 std::cerr << "PARGRID ERROR: parameters vector in constuctor is empty!" << std::endl;
	 return false;
      }
      
      std::map<InputParameter,std::string> zoltanParameters;
      zoltanParameters[imbalanceTolerance] = "IMBALANCE_TOL";
      zoltanParameters[loadBalancingMethod] = "LB_METHOD";
      
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
      
      if (cellWeightsUsed == false && edgeWeightsUsed == false) {
	 std::cerr << "PARGRID ERROR: You must specify cell weight scale, edge weight scale, or both in ";
	 std::cerr << "the first element in parameters vector in constructor!" << std::endl;
	 return false;
      }
      /*
      // TEST
      int counter = 0;
      std::cerr << "LB cell & edge weights:" << std::endl;
      std::cerr << "\t cell weights used: ";
      if (cellWeightsUsed == true) std::cerr << "yes" << std::endl; else std::cerr << "no" << std::endl;
      std::cerr << "\t edge weights used: ";
      if (edgeWeightsUsed == true) std::cerr << "yes" << std::endl; else std::cerr << "no" << std::endl;
      std::cerr << "\t cell weight scale: " << cellWeight << std::endl;
      std::cerr << "\t edge weight scale: " << edgeWeight << std::endl;
      
      for (std::vector<std::list<std::pair<std::string,std::string> > >::const_iterator it=loadBalancingParameters.begin();
	   it!=loadBalancingParameters.end(); ++it) {
	 std::cerr << "LB Parameters for hier level " << counter+1 << std::endl;
	 for (std::list<std::pair<std::string,std::string> >::const_iterator jt=it->begin(); jt!=it->end(); ++jt) {
	    std::cerr << '\t' << jt->first << " = " << jt->second << std::endl;
	 }
	 ++counter;
      }
      // END TEST
      */
      // Create a new Zoltan object and set some initial parameters:
      zoltan = new Zoltan(comm);
      //zoltan->Set_Param("NUM_GLOBAL_PARTS","5");
      zoltan->Set_Param("NUM_GID_ENTRIES","1");
      zoltan->Set_Param("NUM_LID_ENTRIES","1");
      zoltan->Set_Param("LB_APPROACH","PARTITION");
      zoltan->Set_Param("RETURN_LISTS","ALL");
      zoltan->Set_Param("OBJ_WEIGHT_DIM",objWeightDim.c_str());
      zoltan->Set_Param("EDGE_WEIGHT_DIM",edgeWeightDim.c_str());
      zoltan->Set_Param("DEBUG_LEVEL","0");
      zoltan->Set_Param("PHG_CUT_OBJECTIVE","CONNECTIVITY");
      
      // Check if hierarchical partitioning should be enabled:
      if (parameters.size() > 1) {
	 zoltan->Set_Param("LB_METHOD","HIER");
	 zoltan->Set_Param("HIER_CHECKS","0");
	 zoltan->Set_Param("HIER_DEBUG_LEVEL","0");
      } else {
	 for (std::list<std::pair<std::string,std::string> >::const_iterator it=loadBalancingParameters[0].begin();
	      it != loadBalancingParameters[0].end(); ++it) {
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
      //zoltan->Set_Hier_Num_Levels_Fn(getNumberOfHierarchicalLevels,this);   // Hierarchical
      //zoltan->Set_Hier_Part_Fn(getHierarchicalPartNumber,this);
      //zoltan->Set_Hier_Method_Fn(getHierarchicalParameters,this);

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
      currentCell = localCells.end();
      currentCellID = invalid();
      recalculateInteriorCells = true;
      recalculateExteriorCells = true;
      for (typename std::map<unsigned int,Stencil<C> >::iterator it=stencils.begin(); it!=stencils.end(); ++it) {
	 it->second.update();
      }
   }
   
   /** Check if a cell with given global ID exists on this process.
    * In multithreaded mode this function contains a readers-writers lock.
    * @param cellID Global ID of the searched cell.
    * @return If true, the cell exists on this process.
    */
   template<class C>
   bool ParGrid<C>::localCellExists(CellID cellID) {
      // Acquire read access to localCells and search for the given cellID:
      typename std::map<CellID,ParCell<C> >::const_iterator it = localCells.find(cellID);
      if (it == localCells.end()) return false;
      return true;
   }
   
   /** Get a pointer to user-data of the given cell.
    * @param cellID Global ID of the cell that holds the requested user data.
    * @return Pointer to user data, or NULL if a cell with the given global ID does not exist.
    */
   template<class C>
   C* ParGrid<C>::operator[](const CellID& cellID) {
      // Acquire read access to localCells and search for the given cellID:
      typename std::map<CellID,ParCell<C> >::iterator it;
      it = localCells.find(cellID);
      if (it != localCells.end()) return &(it->second.userData);
      it = remoteCells.find(cellID);
      if (it != remoteCells.end()) return &(it->second.userData);
      return NULL;
   }

   /** Reduce given cell's reference count by one.
    * @nbrID Global ID of the cell whose reference count is to be reduced.
    */
   template<class C>
   void ParGrid<C>::reduceReferenceCount(CellID nbrID) {
      --nbrReferences[nbrID];
   }

   template<class C>
   bool ParGrid<C>::startNeighbourExchange(unsigned int stencil,int identifier) {
      if (getInitialized() == false) return false;
      if (identifier == 0) return false;
      if (stencils.find(stencil) == stencils.end()) return false;
      return stencils[stencil].startTransfer(identifier);

      /*
      // First the metadata needs to be exchanged with remote neighbours so that 
      // each process knows how much data it is going to receive:
      counter = 0;
      if (recvRequests.size() < stencils[stencil].recvs.size()) recvRequests.resize(stencils[stencil].recvs.size());
      
      for (std::map<MPI_processID,std::set<CellID> >::const_iterator proc=stencils[stencil].recvs.begin(); proc!=stencils[stencil].recvs.end(); ++proc) {
	 // Count the number of metadata elements:
	 unsigned int size = 0;
	 unsigned int* elements = new unsigned int[proc->second.size()];
	 unsigned int index = 0;
	 for (std::set<CellID>::const_iterator i=proc->second.begin(); i!=proc->second.end(); ++i) {
	    const CellID cellID = *i;
	    elements[index] = remoteCells[cellID].getMetadataElements(identifier);
	    size += elements[index];
	    ++index;
	 }
	 
	 // Create a struct containing all metadata:
	 int* blockLengths = new int[size];
	 MPI_Aint* displacements = new MPI_Aint[size];
	 MPI_Datatype* datatypes = new MPI_Datatype[size];
	 index = 0;
	 unsigned int offset = 0;
	 for (std::set<CellID>::const_iterator i=proc->second.begin(); i!=proc->second.end(); ++i) {
	    const CellID cellID = *i;
	    remoteCells[cellID].getMetadata(identifier,blockLengths+offset,displacements+offset,datatypes+offset);
	    offset += elements[index];
	    ++index;
	 }
	 
	 MPI_Datatype metadata;
	 MPI_Type_create_struct(size,blockLengths,displacements,datatypes,&metadata);
	 
	 // Receive metadata from neighbouring process proc:
	 MPI_Type_commit(&metadata);
	 const int source = proc->first;
	 const int tag    = proc->first;
	 MPI_Irecv(MPI_BOTTOM,1,metadata,source,tag,comm,&(recvRequests[counter]));
	 MPI_Type_free(&metadata);
	 ++counter;
	 
	 delete [] blockLengths; blockLengths = NULL;
	 delete [] displacements; displacements = NULL;
	 delete [] datatypes; datatypes = NULL;
	 delete [] elements; elements = NULL;
      }
      
      counter = 0;
      if (sendRequests.size() < stencils[stencil].sends.size()) sendRequests.resize(stencils[stencil].sends.size());
      for (std::map<MPI_processID,std::set<CellID> >::const_iterator proc=stencils[stencil].sends.begin(); proc!=stencils[stencil].sends.end(); ++proc) {
	 // Count the number of metadata elements:
	 unsigned int size = 0;
	 unsigned int* elements = new unsigned int[proc->second.size()];
	 unsigned int index = 0;	 
	 for (std::set<CellID>::const_iterator i=proc->second.begin(); i!=proc->second.end(); ++i) {
	    const CellID cellID = *i;
	    elements[index] = localCells[cellID].getMetadataElements(identifier);
	    size += elements[index];
	    ++index;
	 }
	 // Create a struct containing all metadata:
	 int* blockLengths = new int[size];
	 MPI_Aint* displacements = new MPI_Aint[size];
	 MPI_Datatype* datatypes = new MPI_Datatype[size];
	 index = 0;
	 unsigned int offset = 0;
	 for (std::set<CellID>::const_iterator i=proc->second.begin(); i!=proc->second.end(); ++i) {
	    const CellID cellID = *i;
	    localCells[cellID].getMetadata(identifier,blockLengths+offset,displacements+offset,datatypes+offset);
	    offset += elements[index];
	    ++index;
	 }
	 
	 MPI_Datatype metadata;
	 MPI_Type_create_struct(size,blockLengths,displacements,datatypes,&metadata);
	 
	 // Send metadata to neighbouring process proc:
	 MPI_Type_commit(&metadata);
	 const int dest = proc->first;
	 const int tag  = myrank;
	 MPI_Isend(MPI_BOTTOM,1,metadata,dest,tag,comm,&(sendRequests[counter]));
	 MPI_Type_free(&metadata);
	 ++counter;
	 
	 delete [] blockLengths; blockLengths = NULL;
	 delete [] displacements; displacements = NULL;
	 delete [] datatypes; datatypes = NULL;
	 delete [] elements; elements = NULL;
      }

      // Wait for metadata transfers to complete:
      MPI_Waitall(N_receives,&(recvRequests[0]),MPI_STATUSES_IGNORE);
      MPI_Waitall(N_sends   ,&(sendRequests[0]),MPI_STATUSES_IGNORE);
      */
      /*
      counter = 0;
      for (std::map<MPI_processID,std::set<CellID> >::const_iterator proc=stencils[stencil].recvs.begin(); proc!=stencils[stencil].recvs.end(); ++proc) {
	 // Count the number of data elements:
	 unsigned int size = 0;
	 unsigned int* elements = new unsigned int[proc->second.size()];
	 unsigned int index = 0;
	 for (std::set<CellID>::const_iterator i=proc->second.begin(); i!=proc->second.end(); ++i) {
	    const CellID cellID = *i;
	    elements[index] = remoteCells[cellID].getDataElements(identifier);
	    size += elements[index];
	    ++index;
	 }
	 
	 // Create a struct containing all data:
	 int* blockLengths = new int[size];
	 MPI_Aint* displacements = new MPI_Aint[size];
	 MPI_Datatype* datatypes = new MPI_Datatype[size];
	 index = 0;
	 unsigned int offset = 0;
	 for (std::set<CellID>::const_iterator i=proc->second.begin(); i!=proc->second.end(); ++i) {
	    const CellID cellID = *i;
	    const bool sending = false;
	    remoteCells[cellID].getData(sending,identifier,blockLengths+offset,displacements+offset,datatypes+offset);
	    offset += elements[index];
	    ++index;
	 }
	 
	 MPI_Datatype data;
	 MPI_Type_create_struct(size,blockLengths,displacements,datatypes,&data);
	 
	 // Receive metadata from neighbouring process proc:
	 MPI_Type_commit(&data);
	 const int source = proc->first;
	 const int tag    = proc->first;
	 MPI_Irecv(MPI_BOTTOM,1,data,source,tag,comm,&(recvRequests[counter]));
	 MPI_Type_free(&data);
	 ++counter;
	 
	 delete [] blockLengths; blockLengths = NULL;
	 delete [] displacements; displacements = NULL;
	 delete [] datatypes; datatypes = NULL;
	 delete [] elements; elements = NULL;
      }

      counter = 0;
      if (sendRequests.size() < stencils[stencil].sends.size()) sendRequests.resize(stencils[stencil].sends.size());
      for (std::map<MPI_processID,std::set<CellID> >::const_iterator proc=stencils[stencil].sends.begin(); proc!=stencils[stencil].sends.end(); ++proc) {
	 // Count the number of data elements:
	 unsigned int size = 0;
	 unsigned int* elements = new unsigned int[proc->second.size()];
	 unsigned int index = 0;
	 for (std::set<CellID>::const_iterator i=proc->second.begin(); i!=proc->second.end(); ++i) {
	    const CellID cellID = *i;
	    elements[index] = localCells[cellID].getDataElements(identifier);
	    size += elements[index];
	    ++index;
	 }
	 // Create a struct containing all data:
	 int* blockLengths = new int[size];
	 MPI_Aint* displacements = new MPI_Aint[size];
	 MPI_Datatype* datatypes = new MPI_Datatype[size];
	 index = 0;
	 unsigned int offset = 0;
	 for (std::set<CellID>::const_iterator i=proc->second.begin(); i!=proc->second.end(); ++i) {
	    const CellID cellID = *i;
	    const bool sending = true;
	    localCells[cellID].getData(sending,identifier,blockLengths+offset,displacements+offset,datatypes+offset);
	    offset += elements[index];
	    ++index;
	 }
	 
	 MPI_Datatype data;
	 MPI_Type_create_struct(size,blockLengths,displacements,datatypes,&data);
	 
	 // Send data to neighbouring process proc:
	 MPI_Type_commit(&data);
	 const int dest = proc->first;
	 const int tag  = myrank;
	 MPI_Isend(MPI_BOTTOM,1,data,dest,tag,comm,&(sendRequests[counter]));
	 MPI_Type_free(&data);
	 ++counter;
	 
	 delete [] blockLengths; blockLengths = NULL;
	 delete [] displacements; displacements = NULL;
	 delete [] datatypes; datatypes = NULL;
	 delete [] elements; elements = NULL;
      }
	 
      // Wait for data transfers to complete:
      MPI_Waitall(N_receives,&(recvRequests[0]),MPI_STATUSES_IGNORE);
      MPI_Waitall(N_sends   ,&(sendRequests[0]),MPI_STATUSES_IGNORE);

      return true;*/
   }
   
   template<class C>
   bool ParGrid<C>::syncCellHosts() {
      if (getInitialized() == false) return false;
      
      // Go through every neighbour of every local cell and check 
      // if this process owns them. If not, then that neighbour is a remote neighbour,
      // and it is added to remoteCells:
      remoteCells.clear();
      for (typename std::map<CellID,ParCell<C> >::const_iterator it=localCells.begin(); it!=localCells.end(); ++it) {
	 for (unsigned char nbr=0; nbr<it->second.info[0]; ++nbr) {
	    const CellID nbrID = it->second.neighbours[nbr];
	    if (nbrID == invalid()) continue;
	    if (localCells.find(nbrID) == localCells.end()) {
	       remoteCells.insert(std::make_pair(nbrID,ParCell<C>()));
	       std::cerr << "P#" << myrank << " C#" << it->first << " remote nbr " << nbrID << std::endl;
	    }
	 }
      }
      
      // Each process sends every other process a list of cells it owns. 
      // We can then figure out remote neighbour hosts from these lists:
      CellID N_cells;
      for (MPI_processID p = 0; p < getProcesses(); ++p) {
	 if (p == getRank()) {
	    // It is this processes turn to send a list of local cells. 
	    // First tell how many cells this process has:
	    N_cells = localCells.size();
	    MPI_Bcast(&N_cells,1,MPI_Type<CellID>(),p,comm);
	    
	    // Create an array containing cell IDs and broadcast:
	    CellID counter = 0;
	    CellID* cells = new CellID[N_cells];
	    for (typename std::map<CellID,ParCell<C> >::const_iterator c=localCells.begin(); c!=localCells.end(); ++c) {
	       cells[counter] = c->first;
	       ++counter;
	    }
	    MPI_Bcast(cells,N_cells,MPI_Type<CellID>(),p,comm);
	    delete [] cells; cells = NULL;
	 } else {
	    // Receive a list of local cells from another process:
	    MPI_Bcast(&N_cells,1,MPI_Type<CellID>(),p,comm);
	    
	    CellID* cells = new CellID[N_cells];
	    MPI_Bcast(cells,N_cells,MPI_Type<CellID>(),p,comm);
	    
	    // Go through the received list and check if any of 
	    // this processes remote neighbours are on that list:
	    typename std::map<CellID,ParCell<C> >::iterator it;
	    for (CellID c=0; c<N_cells; ++c) {
	       it = remoteCells.find(cells[c]);
	       if (it == remoteCells.end()) continue;
	       it->second.host = p;
	    }
	    
	    delete [] cells; cells = NULL;
	 }
      }
      
      // FIXME
      for (typename std::map<CellID,ParCell<C> >::iterator it=localCells.begin(); it!=localCells.end(); ++it) {
	 for (size_t n=0; n<27; ++n) {
	    if (it->second.neighbours[n] == invalid()) {
	       it->second.neighbourHosts[n] = MPI_PROC_NULL;
	       continue;
	    }
	    const CellID nbrID = it->second.neighbours[n];
	    typename std::map<CellID,ParCell<C> >::const_iterator jt = localCells.find(nbrID);
	    if (jt == localCells.end()) jt = remoteCells.find(nbrID);
	    if (jt == remoteCells.end()) {
	       std::cerr << "nbr " << nbrID << " not found in localCells or remoteCells, invalid = " << invalid() << std::endl;
	    }
	    it->second.neighbourHosts[n] = jt->second.host;
	 }
      }
      // END FIXME
      
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
	 it = localCells.find(globalIDs[i]);
	 it->second.userData.getCoordinates(geometryData + i*N_coords);
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
      // Search for the given cell:
      typename std::map<CellID,ParCell<C> >::const_iterator it;
      it = localCells.find(globalID[0]);
      
      // Check that the given cell exists on this process:
      #ifdef DEBUG_PARGRID
         if (it == localCells.end()) {
	    std::cerr << "ParGrid ERROR: getCellCoordinates queried non-existing cell #" << globalID[0] << std::endl;
	    *rcode = ZOLTAN_FATAL;
	    return;
	 }
      #endif

      // Get cell coordinates from user:
      it->second.userData.getCoordinates(geometryData);
      *rcode = ZOLTAN_OK;
   }

   /** Definition for Zoltan callback function ZOLTAN_EDGE_LIST_MULTI_FN. This function is required
    * for graph-based load balancing (GRAPH). The purpose of this function is to tell Zoltan
    * the global IDs of each neighbour of all cells local to this process, as well as the ranks of the 
    * MPI processes who own the neighbours and the weights of the edges (if edge weights are used).
    * @param N_globalIDs Size of global ID.
    * @param N_localIDs Size of local ID.
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
      typename std::map<CellID,ParCell<C> >::const_iterator it;
      typename std::map<CellID,ParCell<C> >::const_iterator nbr;
      
      if (N_weights == 0) {
	 // Edge weights are not calculated
	 for (int i=0; i<N_cells; ++i) {
	    // Find the local cell:
	    const CellID cellID = globalIDs[i];
	    it = localCells.find(cellID);
	 
	    // Copy cell's neighbour information:
	    for (size_t j=0; j<it->second.neighbours.size(); ++j) {
	       const CellID nbrID = it->second.neighbours[j];
	    
	       // Find the neighbour:
	       nbr = localCells.find(nbrID);
	       if (nbr == localCells.end()) nbr = remoteCells.find(nbrID);
	    
	       // Copy neighbour global ID and host process ID:
	       nbrGlobalIDs[counter] = nbrID;
	       nbrHosts[counter]     = nbr->second.host;	    
	       ++counter;
	    }
	 }
      } else {
	 // Edge weights are calculated
	 for (int i=0; i<N_cells; ++i) {
	    // Find the local cell:
	    const CellID cellID = globalIDs[i];
	    it = localCells.find(cellID);
	    
	    // Copy cell's neighbour information:
	    for (size_t j=0; j<it->second.neighbours.size(); ++j) {
	       const CellID nbrID = it->second.neighbours[j];
	       
	       // Find the neighbour:
	       nbr = localCells.find(nbrID);
	       if (nbr == localCells.end()) nbr = remoteCells.find(nbrID);
	       
	       // Copy neighbour global ID and host process ID:
	       nbrGlobalIDs[counter] = nbrID;
	       nbrHosts[counter]     = nbr->second.host;
	       edgeWeights[counter]  = edgeWeight;
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
      typename std::map<CellID,ParCell<C> >::const_iterator it;
      // Attempt to find the queried cell:
      it = localCells.find(globalID[0]);
      
      // Check that the cell exists on this process:
      #ifdef DEBUG_PARGRID
      if (it == localCells.end()) {
	 std::cerr << "ParGrid ERROR: getCellEdges queried non-existing cell #" << globalID[0] << std::endl;
	 *rcode = ZOLTAN_FATAL;
	 return;
      }
      #endif

      // Count global IDs of cell's neighbours into Zoltan structures and calculate 
      // edge weight (if in use):
      int counter = 0;
      typename std::map<CellID,ParCell<C> >::const_iterator nbr;
      if (edgeWeightsUsed == true) {
	 for (size_t i=0; i<it->second.neighbours.size(); ++i) {
	    const CellID nbrID = it->second.neighbours[i];
	    
	    nbr = localCells.find(nbrID);
	    if (nbr == localCells.end()) nbr = remoteCells.find(nbrID);
	    
	    nbrGlobalIDs[counter] = nbrID;
	    nbrHosts[counter]     = nbr->second.host;
	    weight[counter]       = edgeWeight;
	    ++counter;
	 }
      } else {
	 for (size_t i=0; i<it->second.neighbours.size(); ++i) {
	    const CellID nbrID = it->second.neighbours[i];
	    
	    nbr = localCells.find(nbrID);
	    if (nbr == localCells.end()) nbr = remoteCells.find(nbrID);
	    
	    nbrGlobalIDs[counter] = nbrID;
	    nbrHosts[counter]     = nbr->second.host;
	    ++counter;
	 }
      }
      *rcode = ZOLTAN_OK;
   }
   /*
   template<class C>
   void ParGrid<C>::getHierarchicalParameters(void* parGridPtr,int level,Zoltan_Struct* zs,int* rcode) {
      #ifdef DEBUG_PARGRID
         // Sanity check on input parameters:
         if (level < 0 || level > loadBalancingParameters.size()) {*rcode = ZOLTAN_FATAL; return;}
      #endif
      
      // Copy user-defined load balancing parameters to Zoltan structure:
      for (std::list<std::pair<std::string,std::string> >::const_iterator it=loadBalancingParameters[level].begin();
	   it!=loadBalancingParameters[level].end(); ++it) {
	 Zoltan_Set_Param(zs,it->first.c_str(),it->second.c_str());
      }
   } */
   
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

      int edgeCounter = 0;
      int pinCounter = 0;

      // Create list of hyperedges and pins:
      for (typename std::map<CellID,ParCell<C> >::const_iterator it=localCells.begin(); it!=localCells.end(); ++it) {
	 vtxedge_GID[edgeCounter] = it->first;  // The hyperedge has the same global ID as the cell
	 vtxedge_ptr[edgeCounter] = pinCounter; // An index into pin_GID where the pins for this cell are written
	 pin_GID[pinCounter]      = it->first;  // Every cell belong in its own hyperedge

	 // Add pin to this cell and to every existing neighbour:
	 for (size_t i=0; i<it->second.neighbours.size(); ++i) {
	    if (it->second.neighbours[i] == invalid()) continue;
	    pin_GID[pinCounter] = it->second.neighbours[i];
	    ++pinCounter;
	 }
	 ++edgeCounter;
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
      for (typename std::map<CellID,ParCell<C> >::const_iterator it=localCells.begin(); it!=localCells.end(); ++it) {
	 edgeGlobalID[counter] = it->first;
	 edgeWeights[counter]  = edgeWeight;
	 ++counter;
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
      int counter = 0;
      if (cellWeightsUsed == true) {
	 // Iterate over all local cells, and get the cell weights from user. This 
	 // allows support for variable cell weights.
	 for (typename std::map<CellID,ParCell<C> >::const_iterator it=localCells.begin(); it!=localCells.end(); ++it) {
	    globalIDs[counter] = it->first;
	    cellWeights[counter] = it->second.userData.getWeight();
	    ++counter;
	 }
      } else {
	 // Iterate over all local cells and just copy global IDs to Zoltan structures:
	 for (typename std::map<CellID,ParCell<C> >::const_iterator it=localCells.begin(); it!=localCells.end(); ++it) {
	    globalIDs[counter] = it->first;
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
      typename std::map<CellID,ParCell<C> >::const_iterator it;
      for (int i=0; i<N_cells; ++i) {
	 const CellID cellID = globalIDs[i];
	 it = localCells.find(cellID);
	 N_edges[i] = it->second.neighbours.size();
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
      typename std::map<CellID,ParCell<C> >::const_iterator it;
      
      // Attempt to find the queried cell:
      it = localCells.find(globalID[0]);
      
      // Check that the queried cell exists on this process:
      #ifdef DEBUG_PARGRID
      if (it == localCells.end()) {
	 std::cerr << "ParGrid ERROR: getNumberOfEdges queried non-existing cell #" << globalID[0] << std::endl;
	 *rcode = ZOLTAN_FATAL;
	 return 0;
      }
      #endif
      
      // Count the number of neighbours the cell has:
      int edges = it->second.neighbours.size();
      
      // Return the number of edges:
      *rcode = ZOLTAN_OK;
      return edges;
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
      
      // Each local cell has one hyperedge:
      *N_lists = localCells.size();
      
      // Calculate the total number of pins:
      int totalNumberOfPins = 0;
      for (typename std::map<CellID,ParCell<C> >::const_iterator it=localCells.begin(); it!=localCells.end(); ++it) {
	 // Each neighbour of each cell belongs to cell's hyperedge (this includes the cell itself):
	 totalNumberOfPins += it->second.neighbours.size();
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
      *N_edges = localCells.size();
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
      #ifdef DEBUG_PARGRID
         if (isInitialized() != true) {*rcode = ZOLTAN_FATAL; return 0;}
      #endif

      // Get the size of localCells container:
      const int N_localCells = localCells.size();
      *rcode = ZOLTAN_OK;
      return N_localCells;
   }
   
   
}

#endif
