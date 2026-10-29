from math import ceil
from typing import List

from vidur.entities.batch import Batch, Request
from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)


class BatchInProgress:
    """The batch being built for one iteration. Provided; do not modify."""

    def __init__(self) -> None:
        self.requests: List[Request] = []
        self.num_tokens: List[int] = []  # tokens each request adds

    def add(self, request: Request, num_tokens: int) -> None:
        self.requests.append(request)
        self.num_tokens.append(num_tokens)

    @property
    def total_tokens(self) -> int:
        """Tokens already in the batch: pass this to _get_request_next_num_tokens."""
        return sum(self.num_tokens)

    def __len__(self) -> int:
        return len(self.requests)


class SarathiReplicaScheduler(BaseReplicaScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # sarathi config
        self._num_running_batches = 0
        self._preempted_requests = []
        # For vLLM and its derivatives, we only need to set a loose max batch size
        # Memory requirements are handled explicitly by the scheduler
        self._max_micro_batch_size = self._config.batch_size_cap // self._num_stages
        self._watermark_blocks = int(
            self._config.watermark_blocks_fraction * self._config.num_blocks
        )

    def _can_allocate_request(self, request: Request) -> bool:
        if request.id not in self._allocation_map:
            # new request
            num_required_blocks = ceil(
                request.num_prefill_tokens / self._config.block_size
            )
            return (
                self._config.num_blocks
                - self._num_allocated_blocks
                - num_required_blocks
                >= self._watermark_blocks
            )

        # vllm requires at least one block to be available
        return self._config.num_blocks - self._num_allocated_blocks >= 1

    def _allocate_request(self, request: Request) -> None:
        if request.id not in self._allocation_map:
            # new request
            num_required_blocks = ceil(
                request.num_prefill_tokens / self._config.block_size
            )
            self.allocate(request.id, num_required_blocks)
            return

        num_tokens_reserved = self._allocation_map[request.id] * self._config.block_size
        num_tokens_required = max(0, request.num_processed_tokens - num_tokens_reserved)

        assert (
            num_tokens_required == 0 or num_tokens_required == 1
        ), f"num_tokens_required: {num_tokens_required}"

        if num_tokens_required == 0:
            return

        self.allocate(request.id, 1)

    def on_batch_end(self, batch: Batch) -> None:
        self._num_running_batches -= 1

        for request in batch.requests:
            if request.completed:
                self.free(request.id)
            else:
                self._preempted_requests.append(request)

    def _get_request_next_num_tokens(
        self, request: Request, batch_contains_prefill: bool, num_batch_tokens: int
    ) -> int:
        assert not request.completed

        # [Part 3] TODO: how many tokens this request adds to this iteration.
        # Each iteration has a token budget, self._config.chunk_size, and
        # num_batch_tokens tokens are already in it. A decoding request adds
        # one token, always: decode tokens get in even past the budget. The
        # budget limits prompt tokens: a request still on its prompt adds as
        # much of the rest of its prompt as fits in the budget that is left,
        # which can be negative (then 0).
        # Use: request.is_prefill_complete, request.num_prefill_tokens,
        # request.num_processed_tokens, self._config.chunk_size, num_batch_tokens.

        # if decoding, return 1
        if request.is_prefill_complete:
            return 1

        # calculate number of tokens left for the prompt
        remaining_prompt = request.num_prefill_tokens - request.num_processed_tokens
        # calculate number of tokens left to remain within budget
        remaining_budget = self._config.chunk_size - num_batch_tokens
        # return tokens based on remaining prompt and budget tokens 
        if remaining_budget <= 0:
            return 0
        elif remaining_prompt <= remaining_budget:
            return remaining_prompt
        else:
            return remaining_budget

        # raise NotImplementedError("Part 3: fill _get_request_next_num_tokens")
        # END OF YOUR CODE

    def _reserve_kv_or_preempt(self, request: Request) -> bool:
        """Reserve KV memory for one more token of a decoding request.

        If memory is full, preempt the most recently queued request still in
        self._preempted_requests (it restarts from scratch at the front of the
        waiting queue) and try again. If none is left, preempt this request
        itself.

        Returns True if memory is reserved: add the request to the batch.
        Returns False if this request was preempted: leave it out.
        Provided; do not modify.
        """
        while not self._can_allocate_request(request):
            if self._preempted_requests:
                victim = self._preempted_requests.pop(-1)
                victim.restart()
                self.free(victim.id)
                self._request_queue = [victim] + self._request_queue
            else:
                request.restart()
                self.free(request.id)
                self._request_queue = [request] + self._request_queue
                return False
        self._allocate_request(request)
        return True

    # Building one iteration's batch happens in four steps, one function each.
    # self._preempted_requests holds the requests already running (decoding,
    # or partway through their prompt), oldest first. self._request_queue
    # holds requests that have not started, oldest first. The second argument
    # of _get_request_next_num_tokens is unused; pass False.

    def _add_running_decodes(self, batch: BatchInProgress) -> List[Request]:
        """Step 1: add every running request that is decoding.

        Pop requests from the front of self._preempted_requests until it is
        empty or the batch holds self._max_micro_batch_size requests.
          - Still on its prompt (not request.is_prefill_complete): set it
            aside and return it, in order, for step 2.
          - Decoding: call self._reserve_kv_or_preempt(request). False means
            the request was preempted: leave it out. Otherwise add it with
            the tokens _get_request_next_num_tokens gives (one).
        """
        # [Part 3] TODO: step 1.
        prompts = []
        # stopping conditions of the while loop for popping requests
        while self._preempted_requests and (len(batch) < self._max_micro_batch_size):
            # popping requests from the front
            request = self._preempted_requests.pop(0)
            # setting aside the request if still on the prompt and moving on
            if not request.is_prefill_complete:
                prompts.append(request)
                continue
            # if the request was prempted, leave it out and move on
            if not self._reserve_kv_or_preempt(request):
                continue
            # get the number of tokens for this decoding request
            num_tokens = self._get_request_next_num_tokens(request, False, batch.total_tokens)
            # add the request and its token count to the batch
            batch.add(request, num_tokens)

        return prompts

        # raise NotImplementedError("Part 3: fill _add_running_decodes")
        # END OF YOUR CODE

    def _add_unfinished_prompts(
        self, batch: BatchInProgress, prompts: List[Request]
    ) -> List[Request]:
        """Step 2: give each prompt set aside in step 1, in order, as much as
        fits: the tokens _get_request_next_num_tokens gives for the batch so
        far. A prompt that gets 0 is still running: do not add it; return it,
        in order, to keep for the next iteration.
        """
        # [Part 3] TODO: step 2.
        kept = []
        # for each prompt set aside in step 1
        for prompt in prompts:
            # get the number of tokens for the prompt
            num_tokens = self._get_request_next_num_tokens(prompt, False, batch.total_tokens)
            # keep prompt for next iteration if num tokens is 0, else add it to batch
            if num_tokens == 0:
                kept.append(prompt)
            else:
                batch.add(prompt, num_tokens)
        return kept

        # raise NotImplementedError("Part 3: fill _add_unfinished_prompts")
        # END OF YOUR CODE

    def _return_kept_prompts(self, kept: List[Request]) -> None:
        """Step 3: put the prompts kept in step 2 back on
        self._preempted_requests, ahead of the requests still there, then
        sort it by request.arrived_at.
        """
        # [Part 3] TODO: step 3.
        # appending the kept prompts to the beginning of the existing requests
        self._preempted_requests = kept + self._preempted_requests
        # sorting the preempted requests by the request arrival time
        self._preempted_requests.sort(key=lambda request: request.arrived_at)
        # raise NotImplementedError("Part 3: fill _return_kept_prompts")
        # END OF YOUR CODE

    def _add_new_requests(self, batch: BatchInProgress) -> None:
        """Step 4: start new requests from the front of self._request_queue,
        oldest first. Stop, and do not skip to a later request, when
          - len(self._allocation_map) == self._config.batch_size_cap,
          - the batch holds self._max_micro_batch_size requests,
          - not self._can_allocate_request(the oldest waiting request), or
          - it would add 0 tokens.
        Otherwise pop it, reserve its memory with
        self._allocate_request(request), and add it with its tokens.
        """
        # [Part 3] TODO: step 4.
        # while there are requests in the request queue
        while self._request_queue:
            # stop when len(self._allocation_map) == (could be more) self._config.batch_size_cap
            if len(self._allocation_map) >= self._config.batch_size_cap:
                break

            # stop when the batch holds self._max_micro_batch_size requests
            if len(batch) >= self._max_micro_batch_size:
                break

            oldest_request = self._request_queue[0]
            # stop when not self._can_allocate_request(the oldest waiting request)
            if not self._can_allocate_request(oldest_request):
                break

            num_tokens = self._get_request_next_num_tokens(oldest_request, False, batch.total_tokens)
            # stop when the oldest request would add 0 tokens
            if num_tokens == 0:
                break

            # otherwise pop the request
            self._request_queue.pop(0)
            # reserve the memeory
            self._allocate_request(oldest_request)
            # add the request with its tokens
            batch.add(oldest_request, num_tokens)
        # raise NotImplementedError("Part 3: fill _add_new_requests")
        # END OF YOUR CODE

    def _order_waiting_queue(self) -> None:
        """Order self._request_queue before new requests start. Sarathi keeps
        arrival order; SPF (Part 4) overrides this. Provided."""
        return

    def _get_next_batch(self) -> Batch:
        """Build the next iteration's batch. Provided; do not modify."""
        self._order_waiting_queue()
        batch = BatchInProgress()
        prompts = self._add_running_decodes(batch)
        kept = self._add_unfinished_prompts(batch, prompts)
        self._return_kept_prompts(kept)
        self._add_new_requests(batch)
        if not batch.requests:
            return None
        return Batch(self._replica_id, batch.requests, batch.num_tokens)
