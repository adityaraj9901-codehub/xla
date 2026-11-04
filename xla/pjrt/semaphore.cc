/* Copyright 2019 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/pjrt/semaphore.h"

#include <cstdint>
#include <utility>

#include "absl/log/check.h"
#include "absl/synchronization/mutex.h"
#include "tsl/platform/logging.h"

namespace xla {

Semaphore::Semaphore(int64_t capacity)
    : value_(capacity), max_capacity_(capacity) {
  CHECK_GE(capacity, 0);
}

bool Semaphore::CanAcquire(CanAcquireArgs* args) {
  if (args->semaphore->value_ < args->amount) {
    return false;
  }
  for (CanAcquireArgs* w = args->semaphore->head_; w != args; w = w->next) {
    if (w->amount <= args->semaphore->value_) {
      return false;
    }
  }
  return true;
}

void Semaphore::Acquire(int64_t amount) {
  CHECK_GE(amount, 0);

  mu_.lock();
  CanAcquireArgs args;
  args.semaphore = this;
  args.amount = amount;
  args.prev = tail_;
  if (tail_ != nullptr) {
    tail_->next = &args;
  } else {
    head_ = &args;
  }
  tail_ = &args;

  mu_.Await(absl::Condition(&CanAcquire, &args));

  if (args.prev != nullptr) {
    args.prev->next = args.next;
  } else {
    head_ = args.next;
  }
  if (args.next != nullptr) {
    args.next->prev = args.prev;
  } else {
    tail_ = args.prev;
  }
  value_ -= amount;
  mu_.unlock();
}

bool Semaphore::TryAcquire(int64_t amount) {
  CHECK_GE(amount, 0);
  absl::MutexLock lock(mu_);
  if (value_ >= amount) {
    value_ -= amount;
    return true;
  }
  return false;
}

void Semaphore::Release(int64_t amount) {
  CHECK_GE(amount, 0);
  absl::MutexLock lock(mu_);
  value_ += amount;
}

Semaphore::ScopedReservation::~ScopedReservation() {
  if (semaphore_) {
    semaphore_->Release(amount_);
  }
}

Semaphore::ScopedReservation::ScopedReservation(
    ScopedReservation&& other) noexcept
    : semaphore_(std::exchange(other.semaphore_, nullptr)),
      amount_(std::exchange(other.amount_, 0)) {}

Semaphore::ScopedReservation& Semaphore::ScopedReservation::operator=(
    ScopedReservation&& other) noexcept {
  if (this != &other) {
    if (semaphore_) {
      semaphore_->Release(amount_);
    }
    semaphore_ = std::exchange(other.semaphore_, nullptr);
    amount_ = std::exchange(other.amount_, 0);
  }
  return *this;
}

Semaphore::ScopedReservation Semaphore::ScopedAcquire(int64_t amount) {
  Acquire(amount);
  return ScopedReservation(this, amount);
}

}  // namespace xla
