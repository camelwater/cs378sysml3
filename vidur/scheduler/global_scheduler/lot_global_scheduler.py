from typing import List, Tuple

from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler


class LOTGlobalScheduler(BaseGlobalScheduler):
    """Least outstanding tokens (LOT): send each request to the replica with
    the least work in flight, measured in tokens.

    A request's work is the tokens it still has to process:
        request.total_tokens - request.num_processed_tokens
    (the rest of its prompt plus the rest of its output). A replica's work is
    the sum over self.outstanding_requests(replica_id): the requests sent
    there that are still waiting or running.
    """

    def schedule(self) -> List[Tuple[int, Request]]:
        """Assign every request in self._request_queue (oldest first) to a
        replica. Return a list of (replica_id, request) pairs.

        For each request, pick the replica with the least outstanding work;
        on a tie, the lowest replica_id. Then add the request's work to that
        replica right away, so later requests in the same call see it.
        Keep these counts in local variables; do not call
        self.record_assignment, which the simulator calls for every pair
        you return.
        """
        self.sort_requests()
        request_mapping = []
        # [Part 5] TODO: sum each replica's outstanding work, then assign
        # every request in self._request_queue (pop from the front).
        # Replica ids: self._replica_schedulers.keys().

        # calculate sum of work on each replica
        work = {}
        for replica_id in self._replica_schedulers.keys():
            work[replica_id] = 0
            for request in self.outstanding_requests(replica_id):
                work[replica_id] += request.total_tokens - request.num_processed_tokens

        while self._request_queue:
            # remove oldest waiting request
            request = self._request_queue.pop(0)
        
            # find replica with the least remaining work
            replica_id = None
            for id in work:
                # starting with first replica as best choice
                if replica_id is None:
                    replica_id = id
                # preferring a replica with less remaining work
                elif work[id] < work[replica_id]:
                    replica_id = id
                # if a tie is found, prefer the lower replica id
                elif work[id] == work[replica_id] and id < replica_id:
                    replica_id = id
        
            # save for the simulator 
            request_mapping.append((replica_id, request))
            # inclding this request's work when assigning next request
            work[replica_id] += request.total_tokens - request.num_processed_tokens
        
        # raise NotImplementedError("Part 5: fill LOT schedule")
        # END OF YOUR CODE
        return request_mapping
