/** This file is part of ParGrid parallel grid.
 * 
 *  Copyright 2011-2013 Finnish Meteorological Institute
 *
 *  This program is free software: you can redistribute it and/or modify
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

#ifndef PARGRID_COPY_PROTOCOL_MPI_H
#define PARGRID_COPY_PROTOCOL_MPI_H

namespace pargrid {

   namespace mpiprotocol {
      const Real BUFFER_INCREMENT_FACTOR = 1.2;
      const size_t INITIAL_BUFFER_SIZE = 1;
      
      enum tags {
	 BLOCK_SIZES,
	 BUFFER_1ST_COPY,
	 BUFFER_2ND_COPY,
	 SIZE
      };
   }
   
   template<class BUFFER>
   class CopyProtocolMPI {
    public:
      CopyProtocolMPI();
      CopyProtocolMPI(const CopyProtocolMPI& cp);
      ~CopyProtocolMPI();

      bool clear();
      bool set(BUFFER* buffer,bool sender,MPI_Comm comm,pargrid::MPI_processID);
      bool start();
      bool wait();
      
    protected:
      BUFFER* buffer;                          /**< BUFFER that is used in data copy.*/
      MPI_Comm comm;                           /**< MPI communicator that should be used.*/
      MPI_Datatype datatype;                   /**< Derived MPI datatype used to copy buffer contents.*/
      pargrid::MPI_processID myRank;           /**< MPI rank of this process in communicator comm.*/
      size_t partnerBufferSize;                /**< Size of partner's buffer, only significant at sender.*/
      pargrid::MPI_processID partnerRank;      /**< MPI rank of CopyProtocolMPI that this class is paired with.*/
      MPI_Request requests[mpiprotocol::SIZE]; /**< MPI requests used in buffer copy.*/
      bool sender;                             /**< If true, this class is sending buffer contents instead of receiving.*/
      bool setCalled;                          /**< If true, set member function has been successfully called.*/
      bool transferStarted;                    /**< If true, buffer contents are currently being copied.*/
      
      void invalidate();
      
      pargrid::MPI_processID test; // TEST
   };
   
   template<class BUFFER> inline
   CopyProtocolMPI<BUFFER>::CopyProtocolMPI() {
      invalidate();
   }
   
   template<class BUFFER> inline
   CopyProtocolMPI<BUFFER>::CopyProtocolMPI(const CopyProtocolMPI<BUFFER>& cp) {
      invalidate();
   }
   
   template<class BUFFER> inline
   CopyProtocolMPI<BUFFER>::~CopyProtocolMPI() {
      transferStarted = false;
      clear();
   }
   
   template<class BUFFER> inline
   bool CopyProtocolMPI<BUFFER>::clear() {
      if (transferStarted == true) return false;
      
      buffer = NULL;
      //if (comm != MPI_COMM_NULL) MPI_Comm_free(&comm);
      if (datatype != MPI_DATATYPE_NULL) MPI_Type_free(&datatype);
      partnerRank = -1;
      for (int i=0; i<mpiprotocol::SIZE; ++i) requests[i] = MPI_REQUEST_NULL;
      setCalled = false;
      transferStarted = false;

      partnerBufferSize = pargrid::mpiprotocol::INITIAL_BUFFER_SIZE;
      return true;
   }

   template<class BUFFER> inline
   void CopyProtocolMPI<BUFFER>::invalidate() {
      buffer = NULL;
      comm = MPI_COMM_NULL;
      datatype = MPI_DATATYPE_NULL;
      partnerBufferSize = pargrid::mpiprotocol::INITIAL_BUFFER_SIZE;
      partnerRank = -1;
      for (int i=0; i<mpiprotocol::SIZE; ++i) requests[i] = MPI_REQUEST_NULL;
      setCalled = false;
      transferStarted = false;
   }
   
   template<class BUFFER> inline
   bool CopyProtocolMPI<BUFFER>::set(BUFFER* buffer,bool sender,MPI_Comm comm,pargrid::MPI_processID partnerRank) {
      // Cannot set new state if a buffer copy is in progress:
      if (transferStarted == true) return false;
      
      // If set has already been called, clear contents before continuing:
      if (setCalled == true) clear();

      bool success = true;
      if (buffer == NULL) success = false;
      this->buffer = buffer;
      this->sender = sender;

      // Make a copy of the given communicator:
      //MPI_Comm_dup(comm,&(this->comm));
      this->comm = comm;
      MPI_Comm_rank(comm,&myRank);

      // Check that partnerRank is valid:
      int commSize;
      MPI_Comm_size(comm,&commSize);
      this->partnerRank = partnerRank;
      if (partnerRank < 0 || partnerRank >= commSize) success = false;
      if (myRank == partnerRank) success = false;

      // If errors occurred reset communication protocol to invalid state and exit:
      if (success == false) {
	 clear();
	 return success;
      }
      
      // Create MPI datatype for sending a single element in buffer:
      const size_t elementByteSize = buffer->getElementByteSize();      
      MPI_Type_contiguous(elementByteSize,MPI_BYTE,&datatype);
      MPI_Type_commit(&datatype);
      
      // If this protocol is receiving data, resize buffer:
      buffer->setState(false,0);
      if (sender == false) {
	 if (buffer->setBufferSize(pargrid::mpiprotocol::INITIAL_BUFFER_SIZE) == false) success = false;
      }
      partnerBufferSize = pargrid::mpiprotocol::INITIAL_BUFFER_SIZE;
      
      setCalled = true;
      return success;
   }
   
   template<class BUFFER> inline
   bool CopyProtocolMPI<BUFFER>::start() {
      if (setCalled == false) return false;
      if (transferStarted == true) return false;
      
      const size_t N = buffer->getNumberOfBlocks();
      uint32_t* blockSizes = buffer->getBlockSizes();
      char* bufferPointer = reinterpret_cast<char*>(buffer->getBufferPointer());
      const size_t bufferSize = buffer->getBufferSize();
      
      if (sender == true) {

	 const size_t N_total = bufferSize;
	 size_t N_firstMessage = 0;
	 size_t N_secondMessage = 0;
	 if (N_total > partnerBufferSize) {
	    N_firstMessage  = partnerBufferSize;
	    N_secondMessage = N_total - N_firstMessage;
	 } else {
	    N_firstMessage  = N_total;
	    N_secondMessage = 0;
	 }
	 
	 // Write total number of copied elements, and number of
	 // elements copied with first message, to buffer:
	 blockSizes[N+pargrid::buffermetadata::N_ELEMENTS_TOTAL] = N_total;
	 blockSizes[N+pargrid::buffermetadata::N_ELEMENTS_FIRST] = N_firstMessage;

	 MPI_Isend(blockSizes,N+pargrid::buffermetadata::SIZE,MPI_Type<uint32_t>(),partnerRank,pargrid::mpiprotocol::BLOCK_SIZES,comm,&(requests[0]));
	 MPI_Isend(bufferPointer,N_firstMessage,datatype,partnerRank,pargrid::mpiprotocol::BUFFER_1ST_COPY,comm,&(requests[1]));
	 
	 if (N_secondMessage > 0) {
	    const size_t elementByteSize = buffer->getElementByteSize();
	    MPI_Isend(bufferPointer+elementByteSize*N_firstMessage,N_secondMessage,datatype,partnerRank,pargrid::mpiprotocol::BUFFER_2ND_COPY,comm,&(requests[2]));
	    
	    // Increase perceived partner buffer size:
	    partnerBufferSize = static_cast<size_t>(floor(N_total*pargrid::mpiprotocol::BUFFER_INCREMENT_FACTOR));
	 }	 
      } else {
	 MPI_Irecv(blockSizes,N+pargrid::buffermetadata::SIZE,MPI_Type<uint32_t>(),partnerRank,pargrid::mpiprotocol::BLOCK_SIZES,comm,&(requests[0]));
	 MPI_Irecv(bufferPointer,bufferSize,datatype,partnerRank,pargrid::mpiprotocol::BUFFER_1ST_COPY,comm,&(requests[1]));
      }
      
      // Lock buffer:
      buffer->setState(true,0);
      transferStarted = true;
      return true;
   }
   
   template<class BUFFER> inline
   bool CopyProtocolMPI<BUFFER>::wait() {
      if (transferStarted == false) return false;

      MPI_Waitall(3,requests,MPI_STATUSES_IGNORE);
      buffer->setState(false,0); // FIXME
      
      if (sender == false) { // FIXME
	 const size_t N          = buffer->getNumberOfBlocks();
	 uint32_t* blockSizes    = buffer->getBlockSizes();
	 const size_t bufferSize = buffer->getBufferSize();

	 const size_t N_total        = blockSizes[N+pargrid::buffermetadata::N_ELEMENTS_TOTAL];
	 const size_t N_firstMessage = blockSizes[N+pargrid::buffermetadata::N_ELEMENTS_FIRST];

	 if (N_total > bufferSize) { // FIXME
	    // Increase buffer size:
	    const size_t newSize = static_cast<size_t>(floor(N_total*pargrid::mpiprotocol::BUFFER_INCREMENT_FACTOR));
	    buffer->setBufferSize(newSize);
	    
	    // Receive rest of elements:
	    const size_t N_secondMessage = N_total - N_firstMessage;
	    char* bufferPointer = reinterpret_cast<char*>(buffer->getBufferPointer());
	    const size_t elementByteSize = buffer->getElementByteSize();
	    MPI_Irecv(bufferPointer+elementByteSize*N_firstMessage,N_secondMessage,datatype,partnerRank,pargrid::mpiprotocol::BUFFER_2ND_COPY,comm,&(requests[2]));
	    MPI_Waitall(3,requests,MPI_STATUSES_IGNORE);
	 }

	 buffer->setState(false,N_total); // FIXME
      }
      transferStarted = false;
      return true;
   }
      
} // namespace pargrid

#endif
