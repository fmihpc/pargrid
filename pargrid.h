/** This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 * 
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 * 
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

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
   typedef unsigned char NeighbourID;               /**< Datatype ParGrid uses for neighbour type IDs.*/
   typedef unsigned int StencilID;                  /**< Datatype ParGrid uses for stencil IDs.*/
   typedef unsigned int TransferID;                 /**< Datatype ParGrid uses for (data) transfer IDs.*/
   typedef unsigned int DataID;                     /**< Datatype ParGrid uses for user-defined array IDs.*/

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
   
   /** Different partitioning modes that ParGrid supports.
    * These are directly tied to the partitioning modes supported by Zoltan.*/
   enum PartitioningMode {
      partition,                                    /**< New partitioning should always be computed from scratch.*/
      repartition,                                  /**< While repartitioning cells the existing partitioning
						     * should be taken into account to minimize data transfers.*/
      refine                                        /**< Partitioning calls should only slightly improve 
						     * the existing partitioning.*/
   };
   
   const size_t N_neighbours = 27;                  /**< Number of neighbours reserved each parallel cell has (including the cell itself).*/
   const uint32_t ALL_NEIGHBOURS_EXIST = 134217728 - 1; /**< If a cell's neighbour flags field equals this value all its 
							 * neighbours exist, i.e. the cell is an inner cell.*/
   
   // Forward declaration of ParGrid, required for declarations of 
   // auxiliary structs below.
   template<class C> class ParGrid;
   
   // ***************************************************** //
   // ***** DECLARATIONS OF ZOLTAN CALLBACK FUNCTIONS ***** //
   // ***************************************************** //

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

   /** Wrapper for a user-defined parallel data array in ParGrid.
    * The purpose of this struct is to include all information that is 
    * needed to create, delete, and copy parallel arrays. Some helper 
    * functions are also provided.*/
   template<typename C> struct UserDataWrapper {
    public:
      UserDataWrapper();
      UserDataWrapper(const UserDataWrapper& udw);
      ~UserDataWrapper();
      
      void copy(const UserDataWrapper& udw,CellID newCell,CellID oldCell);
      bool finalize();
      void* getAddress();
      void getDatatype(const std::set<CellID>& globalIDs,MPI_Datatype& datatype);
      void getDatatype(const std::map<CellID,CellID>& cellIDs,int* disps,MPI_Datatype& datatype);
      void initialize(ParGrid<C>* pargrid,const std::string& name,size_t N_cells,unsigned int elements,unsigned int byteSize);

      char* array;                   /**< Pointer to array containing the data.*/
      std::string name;              /**< Name of the array.*/
      unsigned int byteSize;         /**< Byte size of a single array element, i.e. sizeof(double) for a double array.*/
      unsigned int N_cells;          /**< Number of cells (=elements) in the array, equal to the number of 
				      * cells (local+remote) on this process.*/
      unsigned int N_elements;       /**< How many values are reserved per cell.*/
      ParGrid<C>* pargrid;           /**< Pointer to ParGrid class that owns this UserDataWrapper.*/
    private:
      MPI_Datatype basicDatatype;    /**< MPI derived datatype that transfers data on a single cell.*/
   };
   
   // ***** PARCELL *****
   
   template<class C> struct ParCell {
    public:
      ParCell();
      ~ParCell();

      C userData;                                  /**< User data.*/

      void getData(bool sending,TransferID transferID,int receivesPosted,int receiveCount,int* blockLengths,MPI_Aint* displacements,MPI_Datatype* types);
      unsigned int getDataElements(TransferID transferID) const;
      void getMetadata(TransferID transferID,int* blockLengths,MPI_Aint* displacements,MPI_Datatype* types);
      unsigned int getMetadataElements(TransferID transferID) const;
      
    private:

   };

   // ***** STENCIL *****
   
   template<class C>
   struct Stencil {
    public:
      Stencil();
      ~Stencil();
      
      bool addTransfer(TransferID transferID,bool recalculate);
      bool addUserDataTransfer(DataID userDataID,TransferID transferID,bool recalculate);
      void clear();
      const std::vector<CellID>& getBoundaryCells() const;
      const std::vector<CellID>& getInnerCells() const;
      bool initialize(ParGrid<C>& pargrid,StencilType stencilType,const std::vector<NeighbourID>& receives);
      bool removeTransfer(TransferID transferID);
      bool startTransfer(TransferID transferID);
      bool update();
      bool wait(TransferID transferID);

    private:
      bool calcLocalUpdateSendsAndReceives();
      bool calcTypeCache(TransferID transferID);

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
	 int N_receives;          /**< Total number of messages received during this transfer.*/
	 int N_sends;             /**< Total number of messages sent during this transfer.*/
	 bool typeVolatile;       /**< If true, MPI Datatypes need to be recalculated every time 
				   * this transfer is started.*/
	 DataID userDataID;       /**< ID of user-defined array. Valid if this transfer synchronizes a user-defined array.*/
	 bool started;            /**< If true, this transfer has started and MPI requests are valid.*/
	 MPI_Request* requests;   /**< MPI requests associated with this transfer.*/
      };

      std::vector<CellID> boundaryCells;                           /**< List of boundary cells of this stencil.*/
      std::vector<CellID> innerCells;                              /**< List of inner cells of this stencil.*/
      bool initialized;                                            /**< If true, Stencil has initialized successfully and is ready for use.*/
      std::vector<NeighbourID> receivedNbrTypeIDs;                 /**< Neighbour type IDs indicating which cells to receive data from.*/
      std::vector<NeighbourID> sentNbrTypeIDs;                     /**< Neighbour type IDs indicating which cells to send data.*/
      std::map<TransferID,std::map<MPI_processID,TypeCache> > typeCaches; /**< MPI datatype caches for each transfer identifier,
									   * one cache per neighbouring process.*/
      std::map<TransferID,TypeInfo> typeInfo;                      /**< Additional data transfer information for each transfer ID.*/
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

      bool addCell(CellID cellID,const std::vector<CellID>& nbrIDs,const std::vector<NeighbourID>& nbrTypes);
      bool addCellFinished();
      StencilID addStencil(pargrid::StencilType stencilType,const std::vector<NeighbourID>& recvNbrTypeIDs);
      bool addTransfer(StencilID stencilID,TransferID transferID,bool recalculate);
      template<typename T> DataID addUserData(std::string& name,unsigned int N_elements);
      bool addUserDataTransfer(DataID userDataID,StencilID stencilID,TransferID transferID,bool recalculate);
      bool balanceLoad();
      void barrier() const;
      void calcNeighbourOffsets(NeighbourID nbrTypeID,int& i_off,int& j_off,int& k_off) const;
      NeighbourID calcNeighbourTypeID(int i_off,int j_off,int k_off) const;
      bool checkPartitioningStatus(int& counter) const;
      bool deleteUserData(DataID userDataID);
      bool finalize();
      const std::vector<CellID>& getBoundaryCells(StencilID stencilID) const;
      CellID* getCellNeighbourIDs(CellID cellID);
      MPI_Comm getComm() const;
      std::vector<CellID>& getExteriorCells();
      const std::vector<CellID>& getGlobalIDs() const;
      const std::vector<MPI_processID>& getHosts() const;
      bool getInitialized() const;
      const std::vector<CellID>& getInnerCells(StencilID stencilID) const;
      std::vector<CellID>& getInteriorCells();
      CellID getLocalID(CellID globalID) const;
      uint32_t getNeighbourFlags(CellID cellID) const;
      const std::set<MPI_processID>& getNeighbourProcesses() const;
      CellID getNumberOfAllCells() const;
      CellID getNumberOfLocalCells() const;
      MPI_processID getProcesses() const;
      MPI_processID getRank() const;
      bool getRemoteNeighbours(CellID cellID,const std::vector<NeighbourID>& nbrTypeIDs,std::vector<CellID>& nbrIDs);
      char* getUserData(DataID userDataID);
      char* getUserData(const std::string& name);
      void getUserDataIDs(std::vector<DataID>& userDataIDs) const;
      bool getUserDataInfo(DataID userDataID,std::string& name,unsigned int& byteSize,unsigned int& N_elements,char*& ptr) const;
      bool getUserDatatype(TransferID transferID,const std::set<CellID>& globalIDs,MPI_Datatype& datatype,bool reverseStencil);
      bool initialize(MPI_Comm comm,const std::vector<std::map<InputParameter,std::string> >& parameters);
      int initPartitioningCounter() const;
      CellID invalid() const;
      CellID invalidCellID() const;
      DataID invalidDataID() const;
      StencilID invalidStencilID() const;
      TransferID invalidTransferID() const;
      bool localCellExists(CellID cellID);
      C* operator[](const CellID& cellID);
      bool setPartitioningMode(PartitioningMode pm);
      bool startNeighbourExchange(StencilID stencilID,TransferID transferID);
      bool wait(StencilID stencilID,TransferID transferID);

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
      std::vector<CellID> cellNeighbours;                                 /**< Local IDs of neighbours of cells, only valid for local cells
									   as remote cells (that are cached on this process) neighbours do 
									   not have local copies.*/
      std::vector<uint32_t> neighbourFlags;                               /**< Neighbour existence flags for local cells.*/
      std::vector<NeighbourID> cellNeighbourTypes;                        /**< Neighbour type IDs of cells' neighbours.*/
      std::vector<UserDataWrapper<C> > userData;                          /**< User-defined data arrays.*/
      std::set<unsigned int> userDataHoles;                               /**< List of holes in userData.*/

      MPI_Comm comm;                                                      /**< MPI communicator used by ParGrid.*/
      CellWeight edgeWeight;                                              /**< Edge weight scale, used to calculate edge weights for Zoltan.*/
      bool edgeWeightsUsed;                                               /**< If true, edge weights are calculated.*/
      bool initialized;                                                   /**< If true, ParGrid initialized correctly and is ready for use.*/
      std::vector<std::list<std::pair<std::string,std::string> > > 
	                                         loadBalancingParameters; /**< LB parameters for Zoltan for each hierarchical level.*/
      MPI_processID myrank;                                               /**< MPI rank of this process in communicator comm.*/
      std::set<MPI_processID> nbrProcesses;                               /**< MPI ranks of neighbour processes. A process is considered 
									   * to be a neighbour if it has any of this processes local 
									   * cells' neighbours. Calculated in balanceLoad.*/
      MPI_processID N_processes;                                          /**< Number of MPI processes in communicator comm.*/
      int partitioningCounter;                                            /**< Counter that is increased every time the mesh is repartitioned.*/
      std::vector<MPI_Request> recvRequests;                              /**< List of MPI requests that are used for receiving data.*/
      std::vector<MPI_Request> sendRequests;                              /**< List of MPI requests that are used for sending data.*/
      std::map<StencilID,Stencil<C> > stencils;                           /**< Transfer stencils, identified by their ID numbers.*/
      Zoltan* zoltan;                                                     /**< Pointer to Zoltan.*/

      #ifdef PROFILE
         int profZoltanLB;
         int profParGridLB;
         int profMPI;
         int profTotalLB;
         int profUserData;
      #endif
      
      bool checkInternalStructures() const;
      void invalidate();
      bool repartitionUserData(size_t N_cells,CellID newLocalsBegin,int N_import,int* importProcesses,const std::map<MPI_processID,unsigned int>& importsPerProcess,
			       int N_export,int* exportProcesses,ZOLTAN_ID_PTR exportLocalIDs,ZOLTAN_ID_PTR exportGlobalIDs);
      bool syncCellHosts();
   };

   // *********************************************************** //
   // ***** USER DATA WRAPPER TEMPLATE FUNCTION DEFINITIONS ***** //
   // *********************************************************** //

   /** Default constructor. Calls UserDataWrapper::finalize().*/
   template<typename T>
   UserDataWrapper<T>::UserDataWrapper() {array=NULL; basicDatatype=MPI_DATATYPE_NULL; finalize();}
   
   /** Copy-constructor. Calls UserDataWrapper::initialize and then copies the data.
    * @param udw UserDataWrapper whose copy is to be made.*/
   template<typename T>
   UserDataWrapper<T>::UserDataWrapper(const UserDataWrapper& udw) {
      initialize(const_cast<ParGrid<T>*>(udw.pargrid),udw.name,udw.N_cells,udw.N_elements,udw.byteSize);
      for (size_t i=0; i<N_cells*N_elements*byteSize; ++i) array[i] = udw.array[i];
   }
   
   /** Destructor. Calls UserDataWrapper::finalize().*/
   template<typename T>
   UserDataWrapper<T>::~UserDataWrapper() {finalize();}

   /** Copy data from given UserDataWrapper.
    * @param udw UserDataWrapper whose data is to be copied.
    * @param newCell Target cell in which data is written.
    * @param oldCell Source cell in udw in which data is read.*/
   template<typename T>
   void UserDataWrapper<T>::copy(const UserDataWrapper& udw,CellID newCell,CellID oldCell) {
      for (unsigned int i=0; i<N_elements*byteSize; ++i) 
	array[newCell*N_elements*byteSize+i] = udw.array[oldCell*N_elements*byteSize+i];
   }
   
   /** Deallocates UserDataWrapper::array and sets internal variables to dummy values.
    * @return If true, finalization completed successfully.*/
   template<typename T>
   bool UserDataWrapper<T>::finalize() {
      //if (basicDatatype != MPI_DATATYPE_NULL) MPI_Type_free(&basicDatatype);
      byteSize = 0;
      name = "";
      pargrid = NULL;
      N_elements = 0;
      N_cells = 0;
      delete [] array; array = NULL;
      return true;
   }

   /** Get a pointer to UserDataWrapper::array.
    * @return Pointer to user data array. NULL value is returned for uninitialized array.*/
   template<typename T>
   void* UserDataWrapper<T>::getAddress() {return array;}

   /** Get a derived MPI datatype that will transfer all the given cells with a single send (or receive).
    * This function calls MPI_Type_commit for the newly created datatype, MPI_Type_free must be called elsewhere.
    * @param globalIDs Global IDs of the cells.
    * @param datatype Variable in which the derived datatype is to be written.*/
   template<typename T>
   void UserDataWrapper<T>::getDatatype(const std::set<CellID>& globalIDs,MPI_Datatype& datatype) {
      // For each global ID, get the corresponding local ID (=index into array) and 
      // store its offset relative to memory address MPI_BOTTOM:
      int* displacements = new int[globalIDs.size()];
      size_t counter = 0;
      for (std::set<CellID>::const_iterator cellGID=globalIDs.begin(); cellGID!=globalIDs.end(); ++cellGID) {
	 const CellID cellLID = pargrid->getLocalID(*cellGID);
	 displacements[counter] = cellLID;
	 ++counter;
      }

      // Create a derived MPI datatype and commit it:
      if (MPI_Type_create_indexed_block(globalIDs.size(),1,displacements,basicDatatype,&datatype) != MPI_SUCCESS) {
	 std::cerr << "(USERDATAWRAPPER) FATAL ERROR: Failed to create MPI datatype!" << std::endl;
	 exit(1);
      }
      MPI_Type_commit(&datatype);
      delete [] displacements; displacements = NULL;
   }
   
   /** Get a derived MPI datatype that will transfer the given cells with a single send (or receive).
    * This function calls MPI_Type_commit for the newly created datatype, MPI_Type_free must be called elsewhere.
    * @param cellIDs List of (global ID, localID) pairs of transferred cells.
    * @param disps Array in which calculated array displacements are to be stored.
    * @param datatype Variable in which the derived datatype is to be written.*/
   template<typename T>
   void UserDataWrapper<T>::getDatatype(const std::map<CellID,CellID>& cellIDs,int* disps,MPI_Datatype& datatype) {
      // For each (globalID,localID) pair store the offset (relative to MPI_BOTTOM) into array to disps:
      size_t counter = 0;
      for (std::map<CellID,CellID>::const_iterator it=cellIDs.begin(); it!=cellIDs.end(); ++it) {
	 disps[counter] = it->second;
	 ++counter;
      }
      
      // Commit the datatype:
      if (MPI_Type_create_indexed_block(cellIDs.size(),1,disps,basicDatatype,&datatype) != MPI_SUCCESS) {
	 std::cerr << "(USERDATAWRAPPER) FATAL ERROR: Failed to create MPI datatype!" << std::endl;
	 exit(1);
      }
      MPI_Type_commit(&datatype);
   }
      
   /** Initialize UserDataWrapper. Memory for UserDataWrapper::array is allocated here.
    * @param pargrid Pointer to ParGrid class that called this function.
    * @param name Name of the user data array.
    * @param N_cells Number of cells in the array, should be equal to the total number of cells 
    * on this process (local + remote).
    * @param N_elements How many values of byte size byteSize are allocated per cell. This makes it 
    * possible to allocate, say, five doubles per cell.
    * @param byteSize Byte size of single value, i.e. sizeof(double) for doubles.*/
   template<typename T>
   void UserDataWrapper<T>::initialize(ParGrid<T>* pargrid,const std::string& name,size_t N_cells,unsigned int N_elements,unsigned int byteSize) {
      this->pargrid = pargrid;
      this->name = name;
      this->byteSize = byteSize;
      this->N_elements = N_elements;
      this->N_cells = N_cells;
      array = new char[N_cells*N_elements*byteSize];
      
      // Create an MPI datatype that transfers a single array element. This makes 
      // it possible to address larger memory spaces.
      MPI_Type_contiguous(N_elements*byteSize,MPI_Type<char>(),&basicDatatype);
      MPI_Type_commit(&basicDatatype);
   }
   
   // *************************************************
   // ***** PARCELL TEMPLATE FUNCTION DEFINITIONS *****
   // *************************************************
   
   template<class C>
   ParCell<C>::ParCell() { }
   
   template<class C>
   ParCell<C>::~ParCell() { }

   template<class C>
   void ParCell<C>::getData(bool sending,TransferID ID,int receivesPosted,int receiveCount,int* blockLengths,MPI_Aint* displacements,MPI_Datatype* types) {
      userData.getData(sending,ID,receivesPosted,receiveCount,blockLengths,displacements,types);
   }
   
   template<class C>
   unsigned int ParCell<C>::getDataElements(TransferID ID) const {
      return userData.getDataElements(ID);
   }
   
   template<class C>
   void ParCell<C>::getMetadata(TransferID ID,int* blockLengths,MPI_Aint* displacements,MPI_Datatype* types) {
      userData.getMetadata(ID,blockLengths,displacements,types);
   }
   
   template<class C>
   unsigned int ParCell<C>::getMetadataElements(TransferID ID) const {
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
      for (typename std::map<unsigned int,TypeInfo>::iterator i=typeInfo.begin(); i!=typeInfo.end(); ++i) {
	 delete [] i->second.requests; i->second.requests = NULL;
      }
   }
   
   template<class C>
   bool Stencil<C>::addTransfer(TransferID transferID,bool recalculate) {
      if (initialized == false) return false;
      if (typeCaches.find(transferID) != typeCaches.end()) return false;
      typeCaches[transferID];
      typeInfo[transferID].typeVolatile = recalculate;
      typeInfo[transferID].requests = NULL;
      typeInfo[transferID].started = false;
      typeInfo[transferID].userDataID = parGrid->invalidDataID();
      calcTypeCache(transferID);
      return true;
   }
   
   template<class C>
   bool Stencil<C>::addUserDataTransfer(DataID userDataID,TransferID transferID,bool recalculate) {
      if (initialized == false) return false;
      if (typeCaches.find(transferID) != typeCaches.end()) {
	 return false;
      }
      typeCaches[transferID];
      typeInfo[transferID].typeVolatile = recalculate;
      typeInfo[transferID].requests = NULL;
      typeInfo[transferID].started = false;
      typeInfo[transferID].userDataID = userDataID;
      calcTypeCache(transferID);
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
	 CellID* nbrIDs = parGrid->getCellNeighbourIDs(i);
	 for (size_t nbr=0; nbr<N_neighbours; ++nbr) {
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
   bool Stencil<C>::calcTypeCache(TransferID transferID) {
      typename std::map<TransferID,TypeInfo>::iterator info = typeInfo.find(transferID);
      if (info == typeInfo.end()) return false;
      
      int* blockLengths       = NULL;
      MPI_Aint* displacements = NULL;
      MPI_Datatype* types     = NULL;

      // Free old datatypes:
      typename std::map<TransferID,std::map<MPI_processID,TypeCache> >::iterator it=typeCaches.find(transferID);
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
	 // Allocate arrays for MPI datatypes:
	 size_t N_recvs = 0;
	 std::map<MPI_processID,std::set<pargrid::CellID> >::const_iterator tmp = recvs.find(jt->first);
	 if (tmp != recvs.end()) N_recvs = tmp->second.size();	 
	 size_t N_sends = 0;
	 tmp = sends.find(jt->first);
	 if (tmp != sends.end()) N_sends = tmp->second.size();
	 
	 if (info->second.userDataID == parGrid->invalidDataID()) {
	    blockLengths = new int[std::max(N_recvs,N_sends)];
	    displacements = new MPI_Aint[std::max(N_recvs,N_sends)];
	    types = new MPI_Datatype[std::max(N_recvs,N_sends)];
	 
	    // Get displacements from cells receiving data:
	    int counter = 0;
	    C dummy;
	    const unsigned int N_dataElements = dummy.getDataElements(transferID);
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
		  (*parGrid)[localID]->getData(sendingData,it->first,-1,-1,blockLengths+counter*N_dataElements,
					       displacements+counter*N_dataElements,types+counter*N_dataElements);
		  break;
		case remoteToLocalUpdates:
		  (*parGrid)[localID]->getData(sendingData,it->first,receivesPosted[*i],recvCounts[*i].size(),blockLengths+counter*N_dataElements,
					       displacements+counter*N_dataElements,types+counter*N_dataElements);
		  ++receivesPosted[*i];
	       break;
	       }
	       ++counter;
	    }
	 
	    // Create MPI datatype for receiving all data at once from process jt->first.
	    // A receive is inserted only if there are data to receive:
	    if (N_recvs > 0) {
	       jt->second.recvs.push_back(TypeWrapper());
	       MPI_Type_create_struct(N_recvs,blockLengths,displacements,types,&(jt->second.recvs.back().type));
	       MPI_Type_commit(&(jt->second.recvs.back().type));
	       ++info->second.N_receives;
	    }
	    
	    // Get displacements from cells sending data:
	    counter = 0;
	    for (std::set<CellID>::const_iterator i=sends[jt->first].begin(); i!=sends[jt->first].end(); ++i) {
	       const CellID localID = parGrid->getLocalID(*i);
	       const bool sendingData = true;
	       const int dummyRecvCount = 0;
	       (*parGrid)[localID]->getData(sendingData,it->first,dummyRecvCount,dummyRecvCount,blockLengths+counter*N_dataElements,
					    displacements+counter*N_dataElements,types+counter*N_dataElements);
	       ++counter;
	    }

	    // Create MPI datatype for sending all data at once to process jt->first.
	    // A send is inserted only if there are data to send:
	    if (N_sends > 0) {
	       jt->second.sends.push_back(TypeWrapper());
	       MPI_Type_create_struct(N_sends,blockLengths,displacements,types,&(jt->second.sends.back().type));
	       MPI_Type_commit(&(jt->second.sends.back().type));
	       ++info->second.N_sends;
	    }
	    delete [] blockLengths; blockLengths = NULL;
	    delete [] displacements; displacements = NULL;
	    delete [] types; types = NULL;
	 } else { // User data array in ParGrid
	    if (N_recvs > 0) {
	       jt->second.recvs.push_back(TypeWrapper());
	       switch (stencilType) {
		case localToRemoteUpdates:
		  parGrid->getUserDatatype(info->second.userDataID,recvs[jt->first],jt->second.recvs.back().type,false);
		  break;
		case remoteToLocalUpdates:
		  break;
	       }
	       ++info->second.N_receives;
	    }

	    if (N_sends > 0) {
	       jt->second.sends.push_back(TypeWrapper());
	       switch (stencilType) {
		case localToRemoteUpdates:
		  parGrid->getUserDatatype(info->second.userDataID,sends[jt->first],jt->second.sends.back().type,false);
		  break;
		case remoteToLocalUpdates:
		  break;
	       }
	       ++info->second.N_sends;
	    }
	 }
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
   bool Stencil<C>::initialize(ParGrid<C>& parGrid,StencilType stencilType,const std::vector<NeighbourID>& receives) {
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
   bool Stencil<C>::removeTransfer(TransferID transferID) {
      // Check that transfer exists:
      if (initialized == false) return false;
      if (typeCaches.find(transferID) == typeCaches.end()) return false;

      // Erase transfer:
      typeCaches.erase(transferID);
      typeInfo.erase(transferID);
      return true;
   }
   
   template<class C>
   bool Stencil<C>::startTransfer(TransferID transferID) {
      typename std::map<TransferID,std::map<MPI_processID,TypeCache> >::iterator it = typeCaches.find(transferID);
      typename std::map<TransferID,TypeInfo>::iterator info = typeInfo.find(transferID);
      if (it == typeCaches.end()) return false;

      if (info->second.typeVolatile == true) calcTypeCache(transferID);
      
      // Post sends and receives:
      unsigned int counter = 0;
      MPI_Request* requests = info->second.requests;
      for (typename std::map<MPI_processID,TypeCache>::iterator proc=it->second.begin(); proc!=it->second.end(); ++proc) {
	 if (info->second.userDataID == parGrid->invalidDataID()) {
	    for (size_t i=0; i<proc->second.recvs.size(); ++i) {
	       MPI_Irecv(MPI_BOTTOM,1,proc->second.recvs[i].type,proc->first,proc->first,parGrid->getComm(),requests+counter);
	       ++counter;
	    }
	    for (size_t i=0; i<proc->second.sends.size(); ++i) {
	       MPI_Isend(MPI_BOTTOM,1,proc->second.sends[i].type,proc->first,parGrid->getRank(),parGrid->getComm(),requests+counter);
	       ++counter;
	    }
	 } else {
	    // TEST
	    void* buffer = parGrid->getUserData(info->second.userDataID);
	    #ifndef NDEBUG
	    if (buffer == NULL) {
	       std::cerr << "(STENCIL) ERROR: User data ID #" << info->second.userDataID << " returned NULL array!" << std::endl;
	       exit(1);
	    }
	    #endif
	    
	    for (size_t i=0; i<proc->second.recvs.size(); ++i) {
	       MPI_Irecv(buffer,1,proc->second.recvs[i].type,proc->first,proc->first,parGrid->getComm(),requests+counter);
	       ++counter;
	    }
	    for (size_t i=0; i<proc->second.sends.size(); ++i) {
	       MPI_Isend(buffer,1,proc->second.sends[i].type,proc->first,parGrid->getRank(),parGrid->getComm(),requests+counter);
	       ++counter;
	    }
	    // END TEST
	 }
      }
      info->second.started = true;
      return true;
   }
   
   template<class C>
   bool Stencil<C>::update() {
      if (initialized == false) return false;
      bool success = calcLocalUpdateSendsAndReceives();
      for (typename std::map<TransferID,std::map<MPI_processID,TypeCache> >::iterator it=typeCaches.begin(); it!=typeCaches.end(); ++it) 
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
   bool Stencil<C>::wait(TransferID transferID) {
      typename std::map<TransferID,TypeInfo>::iterator info = typeInfo.find(transferID);
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
         profUserData  = -1;
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
    * @return If true, the cell was inserted successfully.
    */
   template<class C>
   bool ParGrid<C>::addCell(CellID cellID,const std::vector<CellID>& nbrIDs,const std::vector<NeighbourID>& nbrTypes) {
      if (getInitialized() == false) return false;
      bool success = true;
      
      // Check that the cell doesn't already exist:
      if (global2LocalMap.find(cellID) != global2LocalMap.end()) return false;
      global2LocalMap[cellID] = N_localCells;
      cells.push_back(ParCell<C>());
      hosts.push_back(getRank());
      globalIDs.push_back(cellID);
      
      // Copy cell's neighbours and increase reference count to cell's neighbours:
      const size_t offset = cellNeighbours.size();
      const size_t flagOffset = neighbourFlags.size();
      cellNeighbours.insert(cellNeighbours.end(),N_neighbours,invalid());
      cellNeighbourTypes.insert(cellNeighbourTypes.end(),N_neighbours,calcNeighbourTypeID(0,0,0));
      neighbourFlags.insert(neighbourFlags.end(),0);
            
      uint32_t nbrFlag = (1 << calcNeighbourTypeID(0,0,0));
      for (size_t n=0; n<nbrIDs.size(); ++n) {
	 if (nbrTypes[n] > calcNeighbourTypeID(1,1,1)) success = false;
	 if (nbrIDs[n] == invalid()) success = false;
	 
	 cellNeighbours[offset+nbrTypes[n]]     = nbrIDs[n];
	 cellNeighbourTypes[offset+nbrTypes[n]] = nbrTypes[n];
	 nbrFlag = (nbrFlag | (1 << nbrTypes[n]));
      }
      neighbourFlags[flagOffset] = nbrFlag;
      cellNeighbours[offset+calcNeighbourTypeID(0,0,0)] = cellID;
      
      ++N_localCells;
      return success;
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
	 for (size_t n=0; n<N_neighbours; ++n) {
	    const CellID nbrGID = cellNeighbours[i*N_neighbours+n];
	    if (nbrGID == invalid()) continue;

	    std::map<CellID,CellID>::const_iterator it = global2LocalMap.find(nbrGID);
	    if (it != global2LocalMap.end()) {
	       // Neighbour is a local cell, replace global ID with local ID:
	       const CellID nbrLID = it->second;
	       cellNeighbours[i*N_neighbours+n] = nbrLID;
	    } else {
	       // Neighbour is a remote cell. Insert a local copy of the neighbour, 
	       // and replace global ID with the new local ID:
	       cells.push_back(ParCell<C>());
	       hosts.push_back(MPI_PROC_NULL);
	       globalIDs.push_back(nbrGID);
	       global2LocalMap[nbrGID] = cells.size()-1;
	       cellNeighbours[i*N_neighbours+n] = cells.size()-1;
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
      std::vector<NeighbourID> nbrTypeIDs(27);
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
   StencilID ParGrid<C>::addStencil(pargrid::StencilType stencilType,const std::vector<NeighbourID>& recvNbrTypeIDs) {
      int currentSize = stencils.size();
      if (stencils[currentSize].initialize(*this,stencilType,recvNbrTypeIDs) == false) {
	 stencils.erase(currentSize);
	 currentSize = -1;
      }
      return currentSize;
   }
   
   template<class C>
   bool ParGrid<C>::addTransfer(StencilID stencilID,TransferID transferID,bool recalculate) {
      if (getInitialized() == false) return false;
      typename std::map<StencilID,Stencil<C> >::iterator it = stencils.find(stencilID);
      if (it == stencils.end()) return false;
      return it->second.addTransfer(transferID,recalculate);
   }
   
   template<class C> template<typename T>
   DataID ParGrid<C>::addUserData(std::string& name,unsigned int N_elements) {
      // Check that a user data array with the given name doesn't already exist:
      for (size_t i=0; i<userData.size(); ++i) {
	 if (userData[i].name == name) return invalidDataID();
      }
      
      //unsigned int userDataID = std::numeric_limits<DataID>::max();
      unsigned int userDataID = invalidDataID();
      if (userDataHoles.size() > 0) {
	 userDataID = *userDataHoles.begin();
	 userDataHoles.erase(userDataID);
      } else {
	 userData.push_back(UserDataWrapper<C>());
	 userDataID = userData.size()-1;
      }
      
      // Add a new user data array:
      userData[userDataID].initialize(this,name,cells.size(),N_elements,sizeof(T));
      return userDataID;
   }
   
   /** Add a new MPI transfer for a user-defined data array.
    * @param userDataID ID number of the user data array, the value returned by addUserData function.
    * @param stencilID ID number of the transfer stencil. For a user-defined stencil this is the value 
    * returned by addStencil function. For the default stencil a value 0.
    * @param transferID A unique ID number for the transfer.
    * @param recalculate If true the data array is dynamically reallocated and the MPI datatypes must be 
    * recalculated before starting transfers.
    * @return If true, a new transfer was added successfully.*/
   template<class C>
   bool ParGrid<C>::addUserDataTransfer(DataID userDataID,StencilID stencilID,TransferID transferID,bool recalculate) {
      if (getInitialized() == false) return false;
       if (userDataID >= userData.size() || userDataHoles.find(userDataID) != userDataHoles.end()) return false;
      typename std::map<StencilID,Stencil<C> >::iterator it = stencils.find(stencilID);
      if (it == stencils.end()) return false;
      return it->second.addUserDataTransfer(userDataID,transferID,recalculate);
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
      std::vector<CellID> newCellNeighbours;
      std::vector<NeighbourID> newCellNeighbourTypes;
      std::vector<uint32_t> newNeighbourFlags;      
      newCells.reserve(newCapacity);
      newHosts.reserve(newCapacity);
      newGlobalIDs.reserve(newCapacity);
      newCells.resize(N_newLocalCells);
      newCellNeighbours.reserve(N_neighbours*N_newLocalCells);
      newCellNeighbourTypes.reserve(N_neighbours*N_newLocalCells);
      newNeighbourFlags.reserve(N_newLocalCells);
      
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

      const unsigned int N_dataElements = dummy.getDataElements(0) + 2;

      // Swap all cells' neighbour IDs to global IDs:
      for (CellID c=0; c<N_localCells; ++c) {
	 for (size_t n=0; n<N_neighbours; ++n) {
	    const CellID nbrLID = cellNeighbours[c*N_neighbours+n];
	    if (nbrLID == invalid()) continue;
	    cellNeighbours[c*N_neighbours+n] = globalIDs[nbrLID];
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

	 // Imported cell's neighbour GIDs:
	 MPI_Aint address;
	 MPI_Get_address(&(newCellNeighbours[(newLocalsBegin+counter)*N_neighbours]),&address);
	 blockLengths[procIndex][arrayIndex+0] = N_neighbours;
	 displacements[procIndex][arrayIndex+0] = address;
	 datatypes[procIndex][arrayIndex+0] = MPI_Type<CellID>();
	 
	 // Imported cell's neighbour type IDs:
	 MPI_Get_address(&(newCellNeighbourTypes[(newLocalsBegin+counter)*N_neighbours]),&address);
	 blockLengths[procIndex][arrayIndex+1] = N_neighbours;
	 displacements[procIndex][arrayIndex+1] = address;
	 datatypes[procIndex][arrayIndex+1] = MPI_Type<NeighbourID>();
	 
	 newCells[newLocalsBegin+counter].getData(sending,identifier,0,dummyRecvCount,blockLengths[procIndex]+arrayIndex+2,displacements[procIndex]+arrayIndex+2,datatypes[procIndex]+arrayIndex+2);
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
	 
	 MPI_Aint address;
	 MPI_Get_address(&(cellNeighbours[exportLocalIDs[i]*N_neighbours]),&address);
	 blockLengths[procIndex][arrayIndex+0] = N_neighbours;
	 displacements[procIndex][arrayIndex+0] = address;
	 datatypes[procIndex][arrayIndex+0] = MPI_Type<CellID>();
	 
	 MPI_Get_address(&(cellNeighbourTypes[exportLocalIDs[i]*N_neighbours]),&address);
	 blockLengths[procIndex][arrayIndex+1] = N_neighbours;
	 displacements[procIndex][arrayIndex+1] = address;
	 datatypes[procIndex][arrayIndex+1] = MPI_Type<NeighbourID>();
	 
	 cells[exportLocalIDs[i]].getData(sending,identifier,dummyRecvCount,dummyRecvCount,blockLengths[procIndex]+arrayIndex+2,displacements[procIndex]+arrayIndex+2,datatypes[procIndex]+arrayIndex+2);
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
	 
	 newCellNeighbours.insert(newCellNeighbours.end(),N_neighbours,invalid());
	 newCellNeighbourTypes.insert(newCellNeighbourTypes.end(),N_neighbours,calcNeighbourTypeID(0,0,0));
	 for (size_t n=0; n<N_neighbours; ++n) {
	    newCellNeighbours[counter*N_neighbours+n] = cellNeighbours[c*N_neighbours+n];
	    newCellNeighbourTypes[counter*N_neighbours+n] = cellNeighbourTypes[c*N_neighbours+n];
	 }
	 newNeighbourFlags[counter] = neighbourFlags[c];

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
	 uint32_t nbrFlag = (1 << calcNeighbourTypeID(0,0,0));
	 
	 std::map<CellID,CellID>::const_iterator it;
	 for (size_t n=0; n<N_neighbours; ++n) {
	    const CellID nbrGID = newCellNeighbours[c*N_neighbours+n];
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

	    newCellNeighbours[c*N_neighbours+n] = nbrLID;
	    nbrFlag = (nbrFlag | (1 << newCellNeighbourTypes[c*N_neighbours+n]));
	 }
	 newNeighbourFlags[c] = nbrFlag;
      }
      
      // Repartition user-defined data arrays:
      #ifdef PROFILE
         profile::start("user data",profUserData);
      #endif
      repartitionUserData(newCells.size(),newLocalsBegin,N_import,importProcesses,importsPerProcess,N_export,exportProcesses,exportLocalIDs,exportGlobalIDs);
      #ifdef PROFILE
         profile::stop();
      #endif

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
	 for (size_t n=0; n<N_neighbours; ++n) {
	    const CellID nbrGID = cellNeighbours[exportLID*N_neighbours+n];
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
      cellNeighbours.swap(newCellNeighbours);
      cellNeighbourTypes.swap(newCellNeighbourTypes);
      neighbourFlags.swap(newNeighbourFlags);
      
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
   void ParGrid<C>::barrier() const {
      MPI_Barrier(comm);
   }
   
   template<class C>
   void ParGrid<C>::calcNeighbourOffsets(NeighbourID nbrTypeID,int& i_off,int& j_off,int& k_off) const {
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
    * @return Calculated neighbour type ID.*/
   template<class C>
   NeighbourID ParGrid<C>::calcNeighbourTypeID(int i_off,int j_off,int k_off) const {
      return (k_off+1)*9 + (j_off+1)*3 + i_off+1;
   }

   /** Debugging function, checks ParGrid internal structures for correctness.
    * @return If true, everything is ok.*/
   template<class C>
   bool ParGrid<C>::checkInternalStructures() const {
      if (getInitialized() == false) return false;
      bool success = true;
      std::map<CellID,int> tmpReferences;
      std::set<CellID> tmpRemoteCells;
      
      // Count neighbour references, and collect global IDs of remote neighbours,
      // Also check neighbourFlags for correctness:
      for (size_t cell=0; cell<N_localCells; ++cell) {
	 for (size_t i=0; i<N_neighbours; ++i) {
	    const CellID nbrLID = cellNeighbours[cell*N_neighbours+i];
	    if (nbrLID == invalid()) {
	       if (((neighbourFlags[cell] >> i) & 1) != 0) {
		  std::cerr << "P#" << myrank << " LID#" << cell << " GID#" << globalIDs[cell] << " nbrFlag is one for non-existing nbr type " << i << std::endl;
	       }
	       continue;
	    }
	    if (((neighbourFlags[cell] >> i ) & 1) != 1) {
	       std::cerr << "P#" << myrank << " LID#" << cell << " GID#" << globalIDs[cell] << " nbrFlag is zero for existing nbr type " << i;
	       std::cerr << " nbr LID#" << nbrLID << " GID#" << globalIDs[nbrLID];
	       std::cerr << std::endl;
	    }
		
	    if (nbrLID == cell) continue;
	    ++tmpReferences[nbrLID];
	    if (nbrLID >= N_localCells) tmpRemoteCells.insert(nbrLID);
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
	 for (size_t n=0; n<N_neighbours; ++n) {
	    if (cellNeighbours[i*N_neighbours+n] == invalid()) continue;
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
	    std::cerr << "(PARGRID) ERROR: User-given counter value '" << counter << "' is invalid in checkPartitioningStatus!" << std::endl;
	 }
      #endif
      bool rvalue = false;
      if (counter < partitioningCounter) rvalue = true;
      counter = partitioningCounter;
      return rvalue;
   }
   
   template<class C>
   bool ParGrid<C>::deleteUserData(DataID userDataID) {
      if (userDataID >= userData.size()) return false;
      if (userDataHoles.find(userDataID) != userDataHoles.end()) return false;
      userDataHoles.insert(userDataID);
      return userData[userDataID].finalize();
   }
   
   /** Finalize ParGrid. After this function returns ParGrid cannot be used 
    * without re-initialisation.
    * @return If true, ParGrid finalized successfully.*/
   template<class C>
   bool ParGrid<C>::finalize() {
      if (initialized == false) return false;
      initialized = false;
      stencils.clear();
      delete zoltan; zoltan = NULL;
      return true;
   }
   
   template<class C>
   const std::vector<CellID>& ParGrid<C>::getBoundaryCells(StencilID stencilID) const {
      typename std::map<StencilID,Stencil<C> >::const_iterator it = stencils.find(stencilID);
      #ifndef NDEBUG
         if (it == stencils.end()) {
	    std::cerr << "(PARGRID) ERROR: Non-existing stencil " << stencilID << " requested in getBoundaryCells!" << std::endl;
	    exit(1);
	 }
      #endif
      return it->second.getBoundaryCells();
   }
   
   /** Get cell's neighbours. Non-existing neighbours have their global IDs 
    * set to value ParGrid::invalid().
    * @param localID Local ID of cell whose neighbours are requested.
    * @param Reference to vector containing neihbours' global IDs. Size 
    * of vector is always 27. Vector can be indexed with ParGrid::calcNeighbourTypeID.*/
   template<class C>
   CellID* ParGrid<C>::getCellNeighbourIDs(CellID localID) {
      #ifndef NDEBUG
         if (localID >= N_localCells) {
	    std::cerr << "(PARGRID) ERROR: getCellNeighbourIDs local ID#" << localID << " is too large!" << std::endl;
	    exit(1);
	 }
      #endif
      return &(cellNeighbours[localID*N_neighbours]);
   }
   
   template<class C>
   MPI_Comm ParGrid<C>::getComm() const {return comm;}
   
   template<class C>
   std::vector<CellID>& ParGrid<C>::getExteriorCells() {
      if (recalculateExteriorCells == true) {
	 const unsigned int ALL_EXIST = 134217728 - 1; // This value is 2^27 - 1, i.e. integer with first 27 bits flipped
	 exteriorCells.clear();
	 for (size_t i=0; i<N_localCells; ++i) {
	    if (neighbourFlags[i] != ALL_EXIST) exteriorCells.push_back(i);
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
    * @return If true, ParGrid is ready for use.
    */
   template<class C>
   bool ParGrid<C>::getInitialized() const {return initialized;}

   template<class C>
   const std::vector<CellID>& ParGrid<C>::getInnerCells(StencilID stencilID) const {
      typename std::map<StencilID,Stencil<C> >::const_iterator it = stencils.find(stencilID);
      if (it == stencils.end()) {
	 std::cerr << "(PARGRID) ERROR: Non-existing stencil " << stencilID << " requested in getInnerCells!" << std::endl;
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
      if (it == global2LocalMap.end()) return invalidCellID();
      return it->second;
   }
   
   template<class C>
   uint32_t ParGrid<C>::getNeighbourFlags(CellID localID) const {
      #ifndef NDEBUG
         if (localID >= N_localCells) {
	    std::cerr << "(PARGRID) ERROR: Local ID#" << localID << " too large in getNeighbourFlags!" << std::endl;
	 }
      #endif
      return neighbourFlags[localID];
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
    * @return Number of MPI processes in communicator comm.
    */
   template<class C>
   MPI_processID ParGrid<C>::getProcesses() const {return N_processes;}
   
   /** Get the rank of this process in the MPI communicator used by ParGrid.
    * The value returned by this function is set in ParGrid::initialize.
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
   bool ParGrid<C>::getRemoteNeighbours(CellID localID,const std::vector<NeighbourID>& nbrTypeIDs,std::vector<CellID>& nbrIDs) {
      nbrIDs.clear();
      hosts.clear();
      if (localID >= N_localCells) return false;
      
      // Iterate over given neighbour type IDs and check if this cell has 
      // those neighbours, and if those neighbours are remote:
      for (size_t n=0; n<nbrTypeIDs.size(); ++n) {
	 const NeighbourID nbrType = nbrTypeIDs[n];
	 if (cells[localID].neighbours[nbrType] == invalid()) continue;
	 if (hosts[cells[localID].neighbours[nbrType]] == getRank()) continue;
	 nbrIDs.push_back(cells[localID].neighbours[nbrType]);
      }	 
      return true;
   }
   
   template<class C>
   char* ParGrid<C>::getUserData(DataID userDataID) {
      if (userDataID >= userData.size() || userDataHoles.find(userDataID) != userDataHoles.end()) return NULL;
      return userData[userDataID].array;
   }
   
   template<class C>
   char* ParGrid<C>::getUserData(const std::string& name) {
      for (size_t i=0; i<userData.size(); ++i) {
	 if (userData[i].name == name) return userData[i].array;
      }
      return NULL;
   }
   
   template<class C>
   void ParGrid<C>::getUserDataIDs(std::vector<DataID>& userDataIDs) const {
      userDataIDs.clear();
      for (size_t i=0; i<userData.size(); ++i) {
	 if (userData[i].array == NULL) continue;
	 userDataIDs.push_back(i);
      }
   }
   
   template<class C>
   bool ParGrid<C>::getUserDataInfo(DataID userDataID,std::string& name,unsigned int& byteSize,unsigned int& N_elements,char*& ptr) const {
      if (userDataID >= userData.size() || userDataHoles.find(userDataID) != userDataHoles.end()) return false;
      name = userData[userDataID].name;
      byteSize = userData[userDataID].byteSize;
      N_elements = userData[userDataID].N_elements;
      ptr = userData[userDataID].array;
      return true;
   }
   
   template<class C>
   bool ParGrid<C>::getUserDatatype(DataID userDataID,const std::set<CellID>& globalIDs,MPI_Datatype& datatype,bool reverseStencil) {
      if (userDataID >= userData.size() || userDataHoles.find(userDataID) != userDataHoles.end()) return false;
      userData[userDataID].getDatatype(globalIDs,datatype);
      return true;
   }
   
   /** Initialize ParGrid and Zoltan. Note that MPI_Init must
    * have been called prior to calling this function.
    * @param comm MPI communicator that ParGrid should use.
    * @param parameters Load balancing parameters for all hierarchical levels.
    * The parameters for each hierarchical level are given in a map, whose contents are pairs
    * formed from parameter types and their string values. These maps themselves
    * are packed into a vector, whose first item (map) is used for hierarchical level
    * 0, second item for hierarchical level 1, and so forth. Zoltan is set to use 
    * hierarchical partitioning if vector size is greater than one, otherwise the 
    * load balancing method given in the first element is used.
    * @return If true, ParGrid initialized correctly.*/
   template<class C>
   bool ParGrid<C>::initialize(MPI_Comm comm,const std::vector<std::map<InputParameter,std::string> >& parameters) {
      zoltan = NULL;
      MPI_Comm_dup(comm,&(this->comm));
      
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
      zoltan->Set_Param("REMAP","1");
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
   
   /** Initialize a partitioning counter that can be used to check if mesh 
    * has been repartitioned. This function should only be called during initialization.
    * @return Initial value for partitioning counter.*/
   template<class C>
   int ParGrid<C>::initPartitioningCounter() const {return -1;}
   
   /** Return invalid cell ID. Cell IDs obtained elsewhere may be 
    * tested against this value to see if they are valid. DEPRECATED.
    * @return Invalid cell global ID.*/
   template<class C>
   CellID ParGrid<C>::invalid() const {return std::numeric_limits<CellID>::max();}
   
   /** Return an invalid cell ID.
    * @return Invalid cell (local or global) ID.*/
   template<class C>
   CellID ParGrid<C>::invalidCellID() const {return std::numeric_limits<CellID>::max();}
   
   /** Return an invalid user data ID.
    * @return Invalid user data ID.*/
   template<class C>
   DataID ParGrid<C>::invalidDataID() const {return std::numeric_limits<DataID>::max();}
   
   /** Return an invalid stencil ID.
    * @return Invalid stencil ID.*/
   template<class C>
   StencilID ParGrid<C>::invalidStencilID() const {return std::numeric_limits<StencilID>::max();}
   
   /** Return an invalid transfer ID.
    * @return Invalid transfer ID.*/
   template<class C>
   TransferID ParGrid<C>::invalidTransferID() const {return -1;}
   
   template<class C>
   void ParGrid<C>::invalidate() {
      recalculateInteriorCells = true;
      recalculateExteriorCells = true;
      for (typename std::map<StencilID,Stencil<C> >::iterator it=stencils.begin(); it!=stencils.end(); ++it) {
	 it->second.update();
      }
   }
   
   /** Check if a cell with given global ID exists on this process.
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
   
   template<class C>
   bool ParGrid<C>::repartitionUserData(size_t N_cells,CellID newLocalsBegin,int N_import,int* importProcesses,
					const std::map<MPI_processID,unsigned int>& importsPerProcess,
					int N_export,int* exportProcesses,ZOLTAN_ID_PTR exportLocalIDs,ZOLTAN_ID_PTR exportGlobalIDs) {
      // Create a list of exported cells' (globalID,localID) pairs to each export process:
      std::map<MPI_processID,std::vector<CellID> > transfers;
      for (int i=0; i<N_export; ++i) {
	 if (exportProcesses[i] == getRank()) continue;
	 transfers[exportProcesses[i]].push_back(exportLocalIDs[i]);
      }
      
      // Allocate N_userData MPI_Datatypes for each export process, where 
      // N_userData is the number of currently allocated user data arrays:
      size_t counter = 0;
      const int N_userData = std::max((size_t)0,userData.size()-userDataHoles.size());
      std::vector<std::vector<MPI_Datatype> > userDatatypes;
      for (std::map<MPI_processID,std::vector<CellID> >::const_iterator it=transfers.begin(); it!=transfers.end(); ++it) {
	 if (it->first == getRank()) continue;
	 userDatatypes.push_back(std::vector<MPI_Datatype>(N_userData));
      }
            
      // Allocate a temporary container for new user data. Each array is 
      // initialized to correct size, N_cells = local + remote cells:
      std::vector<UserDataWrapper<C> > newUserData(userData.size());
      for (size_t i=0; i<userData.size(); ++i) {
	 if (userData[i].array == NULL) continue;
	 newUserData[i].initialize(this,userData[i].name,N_cells,userData[i].N_elements,userData[i].byteSize);
      }

      // Allocate enough send and recv requests:
      int N_importProcesses = 0;
      for (std::map<MPI_processID,unsigned int>::const_iterator it=importsPerProcess.begin(); it!=importsPerProcess.end(); ++it) {
	 if (it->first == getRank()) continue;
	 ++N_importProcesses;
      }
      recvRequests.resize(N_userData*N_importProcesses);
      sendRequests.resize(N_userData*transfers.size());

      // Receive user-defined data from import processes:
      counter = 0;
      size_t requestCounter = 0;
      for (size_t data=0; data<newUserData.size(); ++data) {
	 if (newUserData[data].array == NULL) continue;
	 std::map<MPI_processID,MPI_Datatype> importDatatypes;

	 // Create an MPI datatype that transfers a single user data array element:
	 const int byteSize = newUserData[data].N_elements*newUserData[data].byteSize;
	 MPI_Datatype basicDatatype;
	 MPI_Type_contiguous(byteSize,MPI_Type<char>(),&basicDatatype);
	 MPI_Type_commit(&basicDatatype);
	 
	 // Store displacements to newUserdata for each imported cell:
	 counter = newLocalsBegin;
	 std::map<MPI_processID,std::vector<int> > displacements;
	 for (int i=0; i<N_import; ++i) {
	    if (importProcesses[i] == getRank()) continue;
	    displacements[importProcesses[i]].push_back(counter);
	    ++counter;
	 }

	 // Create datatypes for receiving all user data cells at once from each importing process:
	 for (std::map<MPI_processID,std::vector<int> >::iterator it=displacements.begin(); it!=displacements.end(); ++it) {
	    importDatatypes[it->first];
	    MPI_Type_create_indexed_block(it->second.size(),1,&(it->second[0]),basicDatatype,&(importDatatypes[it->first]));
	    MPI_Type_commit(&(importDatatypes[it->first]));
	 }

	 // Receive data:
	 for (std::map<MPI_processID,MPI_Datatype>::iterator it=importDatatypes.begin(); it!=importDatatypes.end(); ++it) {
	    const MPI_processID source = it->first;
	    const int tag              = it->first;
	    void* buffer               = newUserData[data].array;
	    MPI_Irecv(buffer,1,it->second,source,tag,comm,&(recvRequests[requestCounter]));
	    ++requestCounter;
	 }
	 
	 // Free datatypes:
	 for (std::map<MPI_processID,MPI_Datatype>::iterator it=importDatatypes.begin(); it!=importDatatypes.end(); ++it) {
	    MPI_Type_free(&(it->second));
	 }
	 MPI_Type_free(&basicDatatype);
	 importDatatypes.clear();
      }
      
      // Send user-defined data to export processes:
      requestCounter = 0;
      for (size_t data=0; data<userData.size(); ++data) {
	 if (userData[data].array == NULL) continue;
	 std::map<MPI_processID,std::vector<int> > displacements;
	 
	 // Create an MPI datatype that transfers a single user data array element:
	 const int byteSize = userData[data].N_elements*userData[data].byteSize;
	 MPI_Datatype basicDatatype;
	 MPI_Type_contiguous(byteSize,MPI_Type<char>(),&basicDatatype);
	 MPI_Type_commit(&basicDatatype);
	 
	 // Store displacements to userData for each exported cell and send data:
	 for (std::map<MPI_processID,std::vector<CellID> >::const_iterator it=transfers.begin(); it!=transfers.end(); ++it) {
	    for (size_t j=0; j<it->second.size(); ++j) {
	       displacements[it->first].push_back(it->second[j]);
	    }
	    
	    void* buffer       = userData[data].array;
	    MPI_processID dest = it->first;
	    const int tag      = getRank();
	    
	    MPI_Datatype sendDatatype;
	    MPI_Type_create_indexed_block(displacements[it->first].size(),1,&(displacements[it->first][0]),basicDatatype,&sendDatatype);
	    MPI_Type_commit(&sendDatatype);
	    MPI_Isend(buffer,1,sendDatatype,dest,tag,comm,&(sendRequests[requestCounter]));
	    ++requestCounter;
	    MPI_Type_free(&sendDatatype);
	 }
	 
	 // Free datatypes:
	 MPI_Type_free(&basicDatatype);
      }
      
      // Copy data remaining on this process to newUserData:
      for (size_t data=0; data<userData.size(); ++data) {
	 if (userData[data].array == NULL) continue;
	 counter = 0;
	 for (CellID cell=0; cell<N_localCells; ++cell) {
	    if (hosts[cell] != getRank()) continue;
	    newUserData[data].copy(userData[data],counter,cell);
	    ++counter;
	 }
      }
      
      // Wait for data sends and receives to complete:
      MPI_Waitall(recvRequests.size(),&(recvRequests[0]),MPI_STATUSES_IGNORE);
      MPI_Waitall(sendRequests.size(),&(sendRequests[0]),MPI_STATUSES_IGNORE);

      // Swap newUserData and userData:
      userData.swap(newUserData);
      return true;
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
   
   /** Start remote cell data synchronization. 
    * @param stencilID ID of the stencil.
    * @param transferID ID of the data transfer.
    * @return If true, synchronization started successfully.*/
   template<class C>
   bool ParGrid<C>::startNeighbourExchange(StencilID stencilID,TransferID transferID) {
      if (getInitialized() == false) return false;
      if (transferID == 0) return false;
      if (stencils.find(stencilID) == stencils.end()) return false;
      return stencils[stencilID].startTransfer(transferID);
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

   /** Wait for remote cell data synchronization to complete.
    * @param stencilID ID of the stencil.
    * @param transferID ID of the transfer.
    * @return If true, data synchronization completed successfully.*/
   template<class C>
   bool ParGrid<C>::wait(StencilID stencilID,TransferID transferID) {
      if (getInitialized() == false) return false;
      typename std::map<StencilID,Stencil<C> >::iterator sten = stencils.find(stencilID);
      if (sten == stencils.end()) return false;
      return sten->second.wait(transferID);
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
	    for (size_t n=0; n<N_neighbours; ++n) {
	       const CellID nbrLID = cellNeighbours[i*N_neighbours+n];
	       if (nbrLID == invalid()) continue;
	       
	       // Copy neighbour global ID and host process ID:
	       nbrGlobalIDs[counter] = this->globalIDs[nbrLID];
	       nbrHosts[counter]     = hosts[nbrLID];
	       ++counter;
	    }
	 }
      } else {
	 // Edge weights are calculated
	 for (int i=0; i<N_cells; ++i) {
	    // Copy cell's neighbour information:
	    for (size_t n=0; n<N_neighbours; ++n) {
	       const CellID nbrLID = cellNeighbours[localIDs[i]*N_neighbours+n];
	       if (nbrLID == invalid()) continue;
	       
	       // Copy neighbour global ID and host process ID:
	       nbrGlobalIDs[counter] = this->globalIDs[nbrLID];
	       nbrHosts[counter]     = hosts[nbrLID];
	       edgeWeights[counter]  = edgeWeight*cells[nbrLID].userData.getWeight();
	       ++counter;
	    }
	 }	 
      }
      *rcode = ZOLTAN_OK;
   }
   
   /** Definition for Zoltan callback function ZOLTAN_EDGE_LIST_FN. This function is required
    * for graph-based load balancing (GRAPH). The purpose is to give the global IDs of each neighbour of
    * a given cell, as well as the ranks of the MPI processes which have the neighbouring cells.
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
	 for (size_t n=0; n<N_neighbours; ++n) {
	    const CellID nbrLID = cellNeighbours[localID[0]*N_neighbours+n];
	    if (nbrLID == invalid()) continue;
	    nbrGlobalIDs[counter] = globalIDs[nbrLID];
	    nbrHosts[counter]     = hosts[nbrLID];
	    weight[counter]       = edgeWeight;
	    ++counter;
	 }
      } else {
	 for (size_t n=0; n<N_neighbours; ++n) {
	    const CellID nbrLID = cellNeighbours[localID[0]*N_neighbours+n];
	    if (nbrLID == invalid()) continue;
	    nbrGlobalIDs[counter] = globalIDs[nbrLID];
	    nbrHosts[counter]     = cells[nbrLID];
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
      // ------------ TARKISTA TOIMIIKO TÄMÄ ---------------- //
      // ONKO pinGID[pinCounter] OK ?????
      int pinCounter = 0;

      // Create list of hyperedges and pins:
      for (CellID i=0; i<N_localCells; ++i) {
	 vtxedge_GID[i]      = globalIDs[i];
	 vtxedge_ptr[i]      = pinCounter;
	 pin_GID[pinCounter] = globalIDs[i];
	 
	 // Add pin to this cell and to every existing neighbour:
	 for (size_t n=0; n<N_neighbours; ++n) {
	    const CellID nbrLID = cellNeighbours[i*N_neighbours+n];
	    if (nbrLID == invalid()) continue;
	    pin_GID[pinCounter] = globalIDs[nbrLID];
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
    * three-dimensional mesh internally.
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
	 for (size_t n=0; n<N_neighbours; ++n) {
	    if (cellNeighbours[localID*N_neighbours+n] == invalid()) continue;
	    ++edgeSum;
	 }
	 N_edges[i] = edgeSum;
      }
      *rcode = ZOLTAN_OK;
   }
   
   /** Definition of Zoltan callback function ZOLTAN_NUM_EDGES_FN. This function is required
    * for graph-based load balancing (GRAPH). The purpose is to tell how many edges a given cell has, i.e.
    * how many neighbours it has to share data with.
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
      for (size_t n=0; n<N_neighbours; ++n) {
	 if (cellNeighbours[LID*N_neighbours+n] == invalid()) continue;
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
	 for (size_t n=0; n<N_neighbours; ++n) {
	    if (cellNeighbours[i*N_neighbours+n] == invalid()) continue;
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
    * are currently assigned to this process.
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
